# src/robot_control_interface.py
# Description: (MODIFIED) Defines an interface to communicate with Hyundai Robotics controllers.
#              Receives commands via Message Queue, sends commands to robots,
#              and publishes robot status/pose data to Message Queue.

import socket
import json
import time
import threading
import queue # For internal command/status queues
import logging
import uuid # For request-reply correlation IDs

# Import Message Queue client (e.g., pika for RabbitMQ)
try:
    import pika
    pika_available = True
except ImportError:
    pika = None

from src import config # Import the main config module
# from src.data_logger_db import DataLoggerDB # DataLogger passed in __init__

# Setup logging for this module
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') else logging.getLogger(__name__)
if not hasattr(config, 'logging'):
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.INFO)


# --- Message Queue Configuration (from config.py) ---
# MQ_ROBOT_COMMAND_QUEUE = getattr(config, 'MQ_ROBOT_COMMAND_QUEUE', 'robot_commands_queue')
# MQ_ROBOT_RESPONSE_EXCHANGE = getattr(config, 'MQ_ROBOT_RESPONSE_EXCHANGE', 'robot_responses_exchange')
# MQ_ROBOT_STATUS_EXCHANGE = getattr(config, 'MQ_ROBOT_STATUS_EXCHANGE', 'robot_status_exchange')
# MQ_ROBOT_STATUS_ROUTING_KEY_PREFIX = getattr(config, 'MQ_ROBOT_STATUS_ROUTING_KEY_PREFIX', 'robot.status.')


class RobotConnection:
    """
    Manages a single TCP connection to one robot, including sending commands
    and periodically fetching/publishing status.
    (Code from previous advanced version of RobotConnection, with minor adjustments for MQ integration)
    """
    def __init__(self, robot_config, data_logger=None, status_publish_queue=None): # status_publish_queue for RCI
        self.robot_id = robot_config.get('id')
        self.name = robot_config.get('name', f'Robot{self.robot_id}')
        self.host = robot_config.get('ip')
        self.port = robot_config.get('port')
        self.data_logger = data_logger
        self.status_publish_queue = status_publish_queue # Queue to send status to RCI publisher

        self.socket = None
        self.is_connected = False
        self._sequence_id_counter = 0
        self._send_lock = threading.Lock()
        self._receive_lock = threading.Lock() # Protects synchronous receive for commands

        self._latest_status_data = {} # Raw status data
        self._status_lock = threading.Lock() # Protects _latest_status_data
        self._monitoring_thread = None
        self._monitor_running = False
        self._monitor_stop_event = threading.Event()

        logger.info(f"Robot Connection '{self.name}' ({self.host}:{self.port}) initialized.")
        if self.robot_id is None or self.host is None or self.port is None:
             logger.error(f"Robot configuration for '{self.name}' is incomplete: {robot_config}")
             raise ValueError(f"Incomplete robot configuration for {self.name}")

    # ... (connect, disconnect, _cleanup_connection, _send_framed_json, _receive_exactly, _receive_framed_json, _generate_sequence_id methods from previous version) ...
    # Minor change: Ensure logging uses self.name or self.robot_id appropriately.
    # Minor change: Disconnect should also stop_monitoring().

    def connect(self):
        if self.is_connected and self.socket: return True
        logger.info(f"Robot '{self.name}': Attempting connect...")
        # ... (same connect logic as before, ensure data_logger calls use self.data_logger) ...
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(10) # Default for blocking ops
            self.is_connected = True
            logger.info(f"Robot '{self.name}': Connected.")
            if self.data_logger: self.data_logger.log_robot_status(self.robot_id, {"event": "connected", "address": f"{self.host}:{self.port}"})
            return True
        except Exception as e:
            logger.error(f"Robot '{self.name}': Connect error: {e}")
            if self.data_logger: self.data_logger.log_robot_status(self.robot_id, {"event": "connection_failed", "reason": str(e)})
            self._cleanup_connection()
            return False

    def disconnect(self):
        self.stop_monitoring() # Stop monitoring thread first
        if self.is_connected and self.socket:
            logger.info(f"Robot '{self.name}': Disconnecting.")
            # ... (same disconnect logic as before, ensure data_logger calls use self.data_logger) ...
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception as e: logger.error(f"Robot '{self.name}': Error during disconnect: {e}")
            finally: self._cleanup_connection()
            if self.data_logger: self.data_logger.log_robot_status(self.robot_id, {"event": "disconnected"})
        else: logger.debug(f"Robot '{self.name}': Not connected or already disconnected.")


    def send_command_and_wait_response(self, action, parameters=None, timeout=10):
        # ... (This method remains largely the same as the previous advanced version) ...
        # It's used for synchronous command-response interactions.
        # Ensure it uses self._send_lock and self._receive_lock appropriately if called from multiple threads
        # (though RCI will likely serialize commands per robot from its own command queue).
        if not self.is_connected: logger.error(f"Robot '{self.name}': Not connected for cmd '{action}'."); return None
        seq_id = self._generate_sequence_id()
        request = {"action": action, "parameters": parameters or {}, "sequence_id": seq_id}
        logger.info(f"Robot '{self.name}': Sending cmd: '{action}' (Seq ID: {seq_id})")
        if not self._send_framed_json(request): return None
        response = self._receive_framed_json(timeout=timeout) # Uses its own lock and timeout
        # ... (response validation based on seq_id and status as before) ...
        if response and response.get('sequence_id') == seq_id: return response
        elif response: logger.warning(f"Robot '{self.name}': Mismatched/Error response for cmd '{action}' (SeqID {seq_id}): {response}"); return response # Return error response
        return None # Timeout or major receive error

    # --- Real-time Status Monitoring (Modified to use status_publish_queue) ---
    def start_monitoring(self):
        if self._monitor_running: logger.warning(f"Robot '{self.name}': Monitoring already active."); return
        if not self.is_connected: logger.warning(f"Robot '{self.name}': Cannot start monitoring, not connected."); return
        logger.info(f"Robot '{self.name}': Starting status monitor thread.")
        self._monitor_running = True
        self._monitor_stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, name=f'{self.name}-Monitor', daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        if self._monitor_running:
            logger.info(f"Robot '{self.name}': Stopping status monitor thread...")
            self._monitor_running = False
            self._monitor_stop_event.set()
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=getattr(config, 'ROBOT_MONITOR_SHUTDOWN_TIMEOUT_SEC', 0.5))
                if self._monitoring_thread.is_alive(): logger.warning(f"Robot '{self.name}': Monitor thread did not join.")
            self._monitoring_thread = None
            logger.info(f"Robot '{self.name}': Monitor thread stopped.")

    def _monitor_loop(self):
        logger.info(f"Robot '{self.name}': Monitor loop started.")
        poll_interval = getattr(config, 'ROBOT_STATUS_MONITOR_INTERVAL_SEC', 0.05) # 50ms, 20Hz

        while self._monitor_running and not self._monitor_stop_event.is_set():
            loop_start_time = time.time()
            if not self.is_connected: # Check connection status within the loop
                logger.warning(f"Robot '{self.name}': Monitor loop - connection lost. Attempting reconnect...")
                # The main RobotControlInterface's connection manager will handle reconnections.
                # This loop should probably pause or exit if connection is consistently down.
                # For now, just break and let RCI manage it.
                self.stop_monitoring() # Signal self to stop to avoid busy loop on disconnect
                break

            # --- Option 1: Polling (if robot doesn't push status) ---
            # For polling, we send a 'get_status' command and process its response.
            # This uses the command-response mechanism.
            status_response = self.send_command_and_wait_response(action="get_status", timeout=poll_interval * 0.8) # Short timeout

            if status_response and status_response.get('status') == 'success':
                status_data = status_response.get('data', {})
                if status_data:
                    # Add robot_id and timestamp if not present from robot
                    status_data['robot_id'] = self.robot_id
                    status_data['timestamp_utc_polled'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    # Update internal latest status
                    with self._status_lock:
                        self._latest_status_data = status_data
                    # Put data onto the queue for RobotControlInterface to publish
                    if self.status_publish_queue:
                        try:
                            self.status_publish_queue.put(status_data, timeout=0.01) # Non-blocking if possible
                        except queue.Full:
                             logger.warning(f"Robot '{self.name}': Status publish queue is full.")
            elif status_response: # Handle failed 'get_status' or timeout
                 logger.warning(f"Robot '{self.name}': Failed to poll status. Response: {status_response}")
            # else: send_command_and_wait_response returned None (e.g. send/receive failure, connection broken)

            # --- Option 2: Receiving Pushed Data (if robot pushes status unsolicited) ---
            # This would require _receive_framed_json to be called in a loop,
            # similar to AIInferenceEngine or PhysicsInterface's _receive_loop.
            # The dummy server in the previous RCI example simulated push, which is better.
            # If using push, the 'get_status' polling above would be removed.
            # For this example, we stick to polling as it's simpler to show command execution.

            # Control loop frequency
            elapsed = time.time() - loop_start_time
            sleep_duration = poll_interval - elapsed
            if sleep_duration > 0:
                self._monitor_stop_event.wait(timeout=sleep_duration)
        logger.info(f"Robot '{self.name}': Monitor loop finished.")


    def get_latest_status_raw(self): # Renamed to avoid conflict with RCI's public method
        """Retrieves the latest raw status data for this connection."""
        with self._status_lock:
            return self._latest_status_data.copy()

    # send_command_and_wait_response already implemented
    # _send_framed_json, _receive_framed_json, _generate_sequence_id, _cleanup_connection as before.


class RobotControlInterface:
    """
    (MODIFIED) Manages multiple RobotConnection instances.
    Consumes commands from an MQ, executes them, and publishes responses/status to MQ.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs): # Singleton
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_logger=None):
        with self._lock:
            if self._initialized: return
            self._initialized = True

        logger.info("Robot Control Interface (Advanced with MQ) initializing...")
        self.data_logger = data_logger
        self.robot_configs = getattr(config, 'ROBOT_CONFIGS', [])
        self.connections = {} # {robot_id: RobotConnection instance}
        self._status_publish_queue = queue.Queue(maxsize=500) # For status from RobotConnection monitors

        # --- Message Queue Setup ---
        self._mq_connection = None
        self._mq_channel = None # Using one channel for consuming commands and publishing responses/status
        self._mq_lock = threading.Lock()
        self._mq_command_consumer_thread = None
        self._mq_status_publisher_thread = None
        self._stop_mq_event = threading.Event()

        if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
            if not pika: logger.error("Pika library not found. MQ features for RCI disabled.")
            else:
                if self._connect_mq(): logger.info("RCI: MQ Connection established.")
                else: logger.error("RCI: Failed to connect to MQ at initialization.")
        elif config.USE_MESSAGE_QUEUE:
            logger.warning(f"MQ type '{config.MQ_TYPE}' not supported for RCI yet.")


        # Initialize RobotConnection instances
        if self.robot_configs:
            for robot_cfg in self.robot_configs:
                r_id = robot_cfg.get('id')
                if r_id is not None:
                    self.connections[r_id] = RobotConnection(robot_cfg, self.data_logger, self._status_publish_queue)
                else: logger.error(f"Skipping robot config due to missing 'id': {robot_cfg}")

        # Start background thread to publish status from _status_publish_queue to MQ
        if config.USE_MESSAGE_QUEUE and self._mq_channel:
             self._start_status_publisher_thread()


    def _connect_mq(self):
        # ... (Similar MQ connection logic as in AIInferenceEngine and QEA) ...
        # This method should connect and declare:
        # 1. A queue to consume robot commands from (e.g., config.MQ_ROBOT_COMMAND_QUEUE)
        #    - Bind this queue to an exchange if commands are routed.
        # 2. An exchange to publish robot command responses to (e.g., config.MQ_ROBOT_RESPONSE_EXCHANGE)
        # 3. An exchange to publish robot status/pose data to (e.g., config.MQ_ROBOT_STATUS_EXCHANGE)
        if not pika: return False
        with self._mq_lock:
            if self._mq_channel and self._mq_channel.is_open: return True
            try:
                logger.info(f"RCI: Connecting MQ to {config.MQ_HOST}:{config.MQ_PORT}")
                # ... (pika connection parameters and connect) ...
                creds = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                params = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, creds, heartbeat=600)
                self._mq_connection = pika.BlockingConnection(params)
                self._mq_channel = self._mq_connection.channel()

                # Declare command queue (e.g., a direct queue or queue bound to a direct/topic exchange)
                cmd_q = getattr(config, 'MQ_ROBOT_COMMAND_QUEUE', 'robot_commands_q')
                self._mq_channel.queue_declare(queue=cmd_q, durable=True)
                self.robot_command_queue_name = cmd_q # Store for consumer

                # Declare response exchange (e.g., direct or topic for targeted responses)
                resp_ex = getattr(config, 'MQ_ROBOT_RESPONSE_EXCHANGE', 'robot_responses_ex')
                self._mq_channel.exchange_declare(exchange=resp_ex, exchange_type='direct', durable=True)
                self.robot_response_exchange_name = resp_ex

                # Declare status exchange (topic exchange for various status types)
                status_ex = getattr(config, 'MQ_ROBOT_STATUS_EXCHANGE', 'robot_status_ex')
                self._mq_channel.exchange_declare(exchange=status_ex, exchange_type='topic', durable=True)
                self.robot_status_exchange_name = status_ex

                logger.info(f"RCI: MQ connected. CommandQ='{cmd_q}', RespEx='{resp_ex}', StatusEx='{status_ex}'.")
                return True
            except Exception as e:
                logger.error(f"RCI: Failed to connect MQ: {e}", exc_info=True)
                self._disconnect_mq_nolock() # Use no-lock version if called from within lock
                return False

    def _disconnect_mq_nolock(self): # Version for use inside _mq_lock
        if self._mq_channel and self._mq_channel.is_open: try: self._mq_channel.close() except: pass
        if self._mq_connection and self._mq_connection.is_open: try: self._mq_connection.close() except: pass
        self._mq_channel, self._mq_connection = None, None

    def _disconnect_mq(self):
        with self._mq_lock:
            self._disconnect_mq_nolock()
        logger.info("RCI: MQ connection closed.")


    def _publish_to_mq(self, exchange_name, routing_key, message_body_dict, reply_to=None, correlation_id=None):
        # ... (Similar MQ publishing logic as in QEA, using self._mq_channel) ...
        if not (self._mq_channel and self._mq_channel.is_open):
            logger.warning(f"RCI: MQ channel not open for publishing. Attempting reconnect for {routing_key}.")
            if not self._connect_mq(): logger.error(f"Failed to publish to MQ, conn error. Msg for {routing_key} lost."); return False
        try:
            message_body_str = json.dumps(message_body_dict, ensure_ascii=False)
            properties = pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE, content_type='application/json', timestamp=int(time.time()))
            if reply_to: properties.reply_to = reply_to
            if correlation_id: properties.correlation_id = correlation_id

            with self._mq_lock:
                self._mq_channel.basic_publish(exchange=exchange_name, routing_key=routing_key, body=message_body_str, properties=properties)
            return True
        except Exception as e: # Handle connection errors specifically
            logger.error(f"RCI: Failed to publish to MQ. Key: {routing_key}. Error: {e}", exc_info=True)
            if isinstance(e, (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker)): self._disconnect_mq()
            return False


    def _mq_command_callback(self, ch, method, properties, body):
        """Callback for processing robot command messages received from MQ."""
        try:
            command_packet = json.loads(body.decode('utf-8'))
            logger.info(f"RCI MQ Callback: Received command: {command_packet.get('action')} for Robot {command_packet.get('robot_id')}")

            robot_id = command_packet.get('robot_id')
            action = command_packet.get('action')
            params = command_packet.get('parameters')
            correlation_id = properties.correlation_id # For sending response back
            reply_to_queue = properties.reply_to # Queue to send response to

            response_data = {"status": "failure", "error": "Unknown error", "data": None} # Default response

            conn = self.connections.get(robot_id)
            if not conn or not conn.is_connected:
                response_data["error"] = f"Robot {robot_id} not connected or does not exist."
                logger.error(response_data["error"])
            elif action:
                # Execute command using RobotConnection (this is synchronous)
                # Consider using a thread pool for RobotConnection command execution
                # if commands can be long-running and you have many robots.
                # For now, execute directly in the callback thread.
                robot_response = conn.send_command_and_wait_response(action, params) # This is blocking
                if robot_response:
                     response_data = robot_response # Use the robot's actual response structure
                else: # Timeout or major error in send_command_and_wait_response
                     response_data["error"] = f"No valid response or timeout from Robot {robot_id} for action '{action}'."
                     logger.error(response_data["error"])
            else:
                response_data["error"] = "Missing 'action' in command packet."
                logger.error(response_data["error"])

            # Publish response back to the reply_to_queue using correlation_id
            if reply_to_queue and correlation_id:
                response_routing_key = reply_to_queue # Assuming reply_to is the routing key for response
                self._publish_to_mq(self.robot_response_exchange_name, response_routing_key, response_data, correlation_id=correlation_id)
                logger.info(f"RCI: Sent response for CorrID {correlation_id} to {reply_to_queue}: {response_data.get('status')}")
            else:
                logger.warning(f"RCI: No reply_to_queue or correlation_id for command response. Robot: {robot_id}, Action: {action}")

            ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge message processing

        except json.JSONDecodeError:
            logger.error(f"RCI MQ Callback: Failed to decode JSON from command message: {body.decode('utf-8')[:200]}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Discard malformed message
        except Exception as e:
            logger.error(f"RCI MQ Callback: Error processing command message: {e}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Discard on unhandled error


    def _mq_status_publisher_loop(self):
        """Worker thread to get status from _status_publish_queue and publish to MQ."""
        logger.info("RCI MQ Status Publisher thread started.")
        status_key_prefix = getattr(config, 'MQ_ROBOT_STATUS_ROUTING_KEY_PREFIX', 'robot.status.')

        while not self._stop_mq_event.is_set():
            try:
                robot_status_data = self._status_publish_queue.get(timeout=1.0) # Wait 1 sec

                robot_id = robot_status_data.get('robot_id', 'unknown')
                # Example routing key: "robot.status.robot1.pose" or "robot.status.robot1.full"
                # Depending on how granular the status updates are.
                # Let's publish the full status under a generic key per robot.
                routing_key = f"{status_key_prefix}robot{robot_id}"
                if not self._publish_to_mq(self.robot_status_exchange_name, routing_key, robot_status_data):
                     logger.warning(f"RCI: Failed to publish status for Robot {robot_id} to MQ. Re-queueing (simple retry).")
                     try: self._status_publish_queue.put(robot_status_data, timeout=0.1)
                     except queue.Full: logger.error("RCI: Status publish re-queue failed, queue full.")

                self._status_publish_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"RCI MQ Status Publisher error: {e}", exc_info=True)
                time.sleep(0.5)
        logger.info("RCI MQ Status Publisher thread stopped.")


    def start_interface(self):
        """Starts MQ command consumer, status publisher, and connects to all robots."""
        if not self._initialized_connections:
             logger.error("RCI: Robot connections not initialized. Cannot start interface.")
             return

        logger.info("Robot Control Interface starting...")
        self._stop_mq_event.clear()

        # 1. Connect to MQ (if not already connected)
        if config.USE_MESSAGE_QUEUE and not (self._mq_channel and self._mq_channel.is_open):
            if not self._connect_mq():
                logger.error("RCI: Failed to connect to MQ. Interface may not function fully.")
                # Decide if this is a fatal error for RCI startup

        # 2. Start MQ Command Consumer Thread
        if config.USE_MESSAGE_QUEUE and self._mq_channel:
            self._mq_command_consumer_thread = threading.Thread(target=self._mq_command_consumer_loop, name="RCIMQCommandConsumer", daemon=True)
            self._mq_command_consumer_thread.start()
        else:
             logger.warning("RCI: MQ not enabled or not connected. Command consumption via MQ disabled.")


        # 3. Connect to Robots (starts their individual monitoring threads)
        # connect_all now returns list of successfully connected robot_ids
        self.connect_all() # This also updates self.connected_robot_ids
        if self.connected_robot_ids:
             # Start monitoring for all connected robots (RobotConnection's _monitor_loop polls and puts to _status_publish_queue)
             self.start_monitoring_all_robots()
        else:
             logger.warning("RCI: No robots connected after connect_all attempt.")


    def _mq_command_consumer_loop(self):
        """Main loop for the MQ command consumer thread."""
        logger.info("RCI MQ command consumer loop started.")
        while not self._stop_mq_event.is_set():
            try:
                if not (self._mq_channel and self._mq_channel.is_open):
                    logger.warning("RCI: MQ channel not open in command consumer loop. Reconnecting...")
                    if not self._connect_mq(): time.sleep(5); continue

                logger.info(f"RCI: Starting to consume robot commands from MQ queue '{self.robot_command_queue_name}'...")
                # Set prefetch_count=1 for fair dispatch if multiple RCI instances (or workers) consume from same queue
                # self._mq_channel.basic_qos(prefetch_count=1)
                self._mq_channel.basic_consume(queue=self.robot_command_queue_name, on_message_callback=self._mq_command_callback) # auto_ack=False by default if not specified
                self._mq_channel.start_consuming() # Blocking call
                logger.info("RCI: MQ command consumption stopped.")
                if self._stop_mq_event.is_set(): break
            # ... (similar MQ exception handling as in QEA and AIE consumer loops) ...
            except pika.exceptions.ConnectionClosedByBroker: logger.warning("RCI: MQ Connection (Cmd) closed by broker. Reconnecting..."); self._disconnect_mq(); time.sleep(5)
            except Exception as e: logger.error(f"RCI: MQ command consumer loop error: {e}", exc_info=True); self._disconnect_mq(); time.sleep(5)
        logger.info("RCI MQ command consumer loop stopped.")

    def _start_status_publisher_thread(self):
        if self._mq_status_publisher_thread is None or not self._mq_status_publisher_thread.is_alive():
            self._mq_status_publisher_thread = threading.Thread(target=self._mq_status_publisher_loop, name="RCIMQStatusPublisher", daemon=True)
            self._mq_status_publisher_thread.start()

    def stop_interface(self):
        """Stops MQ threads, robot monitoring, and disconnects from robots and MQ."""
        logger.info("Robot Control Interface stopping...")
        self._stop_mq_event.set() # Signal all MQ related threads to stop

        # Stop robot monitoring first (RobotConnection threads)
        self.stop_monitoring_all_robots()

        # Stop MQ Command Consumer Thread
        if self._mq_command_consumer_thread and self._mq_command_consumer_thread.is_alive():
            if self._mq_channel and self._mq_channel.is_open: try: self._mq_channel.stop_consuming() except: pass
            logger.info("Waiting for RCI MQ Command Consumer thread...")
            self._mq_command_consumer_thread.join(timeout=5.0)
            if self._mq_command_consumer_thread.is_alive(): logger.warning("RCI MQ Command Consumer thread did not join.")
        self._mq_command_consumer_thread = None

        # Stop MQ Status Publisher Thread
        if self._mq_status_publisher_thread and self._mq_status_publisher_thread.is_alive():
            logger.info("Waiting for RCI MQ Status Publisher thread...")
            self._mq_status_publisher_thread.join(timeout=5.0) # It checks stop_event from queue timeout
            if self._mq_status_publisher_thread.is_alive(): logger.warning("RCI MQ Status Publisher thread did not join.")
        self._mq_status_publisher_thread = None

        # Disconnect from Robots
        self.disconnect_all()

        # Disconnect from MQ
        self._disconnect_mq()
        logger.info("Robot Control Interface stopped.")


    # --- Public Methods (Wrappers for MQ or direct calls if MQ disabled) ---
    # These methods are now for direct invocation by SystemManager or other *local* Python code
    # IF MQ is NOT used. If MQ is used, commands should be sent to MQ_ROBOT_COMMAND_QUEUE.
    # For clarity, let's assume these direct call methods are for fallback or testing.

    def get_latest_robot_pose(self, robot_id): # Maintained for WPM's AC loop direct access
        conn = self.connections.get(robot_id)
        if conn and conn.is_connected:
            raw_status = conn.get_latest_status_raw()
            if raw_status:
                 # Extract pose (this needs to align with what _monitor_loop stores)
                 pose = {}
                 if 'joint_angles' in raw_status: pose['joint_angles'] = raw_status['joint_angles']
                 if 'position_tcp' in raw_status: pose['tcp_transform'] = {'position': raw_status['position_tcp'], 'rotation': raw_status.get('orientation_tcp')}
                 return pose if pose else None
        return None

    # ... (Other direct call methods like run_job, stop_job from previous version can be kept for testing/fallback) ...
    # ... (Or they are removed if all control is strictly via MQ commands) ...
    # For this example, we'll assume WPM sends commands to MQ, so these direct methods are less critical for main flow.

    def start_monitoring_all_robots(self): # Helper called by start_interface
        for robot_id in self.get_connected_robot_ids(): # Use current list of connected
             conn = self.connections.get(robot_id)
             if conn: conn.start_monitoring()

    def stop_monitoring_all_robots(self): # Helper called by stop_interface
        for robot_id, conn in self.connections.items():
             if conn: conn.stop_monitoring()


# Example Usage (RCI is usually managed by SystemManager)
if __name__ == '__main__':
    # ... (Example usage needs significant updates to test MQ command consumption and status publishing) ...
    # ... (It would involve setting up a dummy MQ publisher for commands, and a dummy MQ subscriber for status/responses) ...
    logger.warning("RobotControlInterface is designed to be orchestrated by SystemManager. Direct run example is complex for MQ setup.")
    logger.info("To test, ensure RabbitMQ is running and config.py has correct MQ/Robot settings.")
    # A simple test could be to initialize, connect to dummy robots, and check if monitoring pushes to status queue.