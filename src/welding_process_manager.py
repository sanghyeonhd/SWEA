# src/welding_process_manager.py
# Description: (MODIFIED) Manages the overall welding workflow, using Message Queues
#              for coordination with other modules (robots, AI, evaluation, UE viz).

import time
import logging
import threading
import enum
import queue # For internal processing or specific response matching
import json
import uuid # For correlation IDs in MQ request-reply

# Import Message Queue client (e.g., pika for RabbitMQ)
try:
    import pika
    pika_available = True
except ImportError:
    pika = None

from src import config
# Dependencies (instances injected by SystemManager)
# from src.robot_control_interface import RobotControlInterface # Methods are now primarily via MQ
# from src.ai_inference_engine import AIInferenceEngine # Predictions received via MQ
# from src.quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl # Evaluations received via MQ
# from src.sensor_data_handler import SensorDataHandler # Sensor data mainly goes to AI engine via MQ
# from src.data_logger_db import DataLoggerDB
# from src.physics_interface import UnrealSimulatorInterface # Still direct calls for visualization

# Setup logging
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') else logging.getLogger(__name__)
if not hasattr(config, 'logging'):
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.INFO)


# --- Process States (same as before) ---
class WeldingProcessState(enum.Enum):
    IDLE = 0; LOADING_PART = 1; ROBOT_MOVING_TO_START = 2; WELDING_IN_PROGRESS = 3
    ADAPTIVE_CONTROL_ACTIVE = 4; WELDING_COMPLETED = 5; ROBOT_MOVING_TO_HOME = 6
    UNLOADING_PART = 7; ERROR = 8; ABORTED = 9; WAITING_ROBOT_RESPONSE = 10
    WAITING_AI_PREDICTION = 11; WAITING_QUALITY_EVALUATION = 12


# --- Welding Job/Recipe (load from config or DB) ---
WELDING_JOBS_RECIPES = config.load_json_config(getattr(config, 'WELDING_RECIPES_PATH', 'config/welding_recipes.json')) or {}
if not WELDING_JOBS_RECIPES:
    logger.warning("Welding recipes not found or empty. Process Manager will not be able to run jobs.")


class WeldingProcessManager:
    """
    (MODIFIED) Manages the welding workflow using Message Queues for inter-module communication.
    """
    _instance = None; _lock = threading.Lock()
    def __new__(cls, *args, **kwargs): # Singleton
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_logger=None, physics_interface=None, robot_control_interface_direct_access=None):
        # robot_control_interface_direct_access is for things like get_latest_pose if not fully on MQ
        with self._lock:
            if self._initialized: return
            self._initialized = True

            logger.info("Welding Process Manager (Advanced with MQ) initializing...")
            self.data_logger = data_logger
            self.physics_interface = physics_interface # For direct UE visualization commands
            self.robot_interface_direct = robot_control_interface_direct_access # For get_latest_pose

            self.current_job_id = None
            self.current_job_recipe = None
            self.current_process_state = WeldingProcessState.IDLE
            self.active_robots_state = {} # {robot_id: {"status": "...", "current_params": {...}, "last_response": ...}}
            self.process_thread = None
            self.stop_event = threading.Event()
            self._adaptive_control_threads = {}

            # --- MQ Setup ---
            self._mq_connection = None
            self._mq_channel = None
            self._mq_lock = threading.Lock()
            self._mq_consumer_thread = None # For consuming responses and status updates
            # For request-reply pattern, store events to wait on for specific correlation_ids
            self._mq_response_events = {} # {correlation_id: (threading.Event(), response_data_holder)}
            self.response_queue_name = None # Unique queue for receiving replies

            if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
                if not pika: logger.error("Pika library not found. MQ features for WPM disabled.")
                else:
                    if self._connect_mq(): logger.info("WPM: MQ Connection established.")
                    else: logger.error("WPM: Failed to connect to MQ at initialization.")
            elif config.USE_MESSAGE_QUEUE:
                logger.warning(f"MQ type '{config.MQ_TYPE}' not supported for WPM yet.")

    def _connect_mq(self):
        # ... (Similar MQ connection logic as in other advanced modules) ...
        # This method should connect and declare:
        # 1. An exchange to publish robot commands to (e.g., config.MQ_ROBOT_COMMAND_EXCHANGE - if RCI uses an exchange)
        #    OR a direct queue name (config.MQ_ROBOT_COMMAND_QUEUE) if RCI consumes from a specific queue.
        # 2. An exchange/queue to publish AI prediction requests to (if AIE consumes requests).
        # 3. An exchange/queue to publish Quality Evaluation requests to (if QEA consumes requests).
        # 4. An exclusive queue to receive replies for its requests (for request-reply pattern).
        # 5. Bind this reply queue to exchanges where responses are published (e.g., config.MQ_ROBOT_RESPONSE_EXCHANGE).
        # 6. Optionally, subscribe to general robot status updates from config.MQ_ROBOT_STATUS_EXCHANGE.
        if not pika: return False
        with self._mq_lock:
            if self._mq_channel and self._mq_channel.is_open: return True
            try:
                logger.info(f"WPM: Connecting MQ to {config.MQ_HOST}:{config.MQ_PORT}")
                # ... (pika connection parameters and connect) ...
                creds = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                params = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, creds, heartbeat=600)
                self._mq_connection = pika.BlockingConnection(params)
                self._mq_channel = self._mq_connection.channel()

                # Declare exchanges this module publishes to
                self._mq_channel.exchange_declare(exchange=getattr(config, 'MQ_ROBOT_COMMAND_EXCHANGE', 'robot_commands_ex'), exchange_type='direct', durable=True)
                self._mq_channel.exchange_declare(exchange=getattr(config, 'MQ_AI_REQUEST_EXCHANGE', 'ai_requests_ex'), exchange_type='direct', durable=True)
                # QEA might listen for AI predictions directly or WPM forwards them

                # Declare an exclusive queue for receiving replies (request-reply pattern)
                result = self._mq_channel.queue_declare(queue='', exclusive=True)
                self.response_queue_name = result.method.queue
                # Bind this reply queue to exchanges where responses are published by other services
                # Example: Bind to robot response exchange
                robot_resp_ex = getattr(config, 'MQ_ROBOT_RESPONSE_EXCHANGE', 'robot_responses_ex')
                self._mq_channel.exchange_declare(exchange=robot_resp_ex, exchange_type='direct', durable=True)
                self._mq_channel.queue_bind(exchange=robot_resp_ex, queue=self.response_queue_name, routing_key=self.response_queue_name) # Use queue name as routing key for direct replies

                # Example: Bind to AI prediction exchange if WPM needs to see all predictions
                ai_pred_ex = config.MQ_AI_PREDICTION_EXCHANGE
                self._mq_channel.exchange_declare(exchange=ai_pred_ex, exchange_type='topic', durable=True)
                # self._mq_channel.queue_bind(exchange=ai_pred_ex, queue=self.response_queue_name, routing_key="ai.prediction.#") # Consume all AI predictions

                # Example: Bind to QEA results exchange
                qea_res_ex = getattr(config, 'MQ_QEA_RESULTS_EXCHANGE', 'qea_results_exchange')
                self._mq_channel.exchange_declare(exchange=qea_res_ex, exchange_type='topic', durable=True)
                # self._mq_channel.queue_bind(exchange=qea_res_ex, queue=self.response_queue_name, routing_key="qea.evaluation.#") # Consume all evaluations

                logger.info(f"WPM: MQ Connected. ReplyQ='{self.response_queue_name}'.")
                return True
            except Exception as e:
                logger.error(f"WPM: Failed to connect MQ: {e}", exc_info=True)
                self._disconnect_mq_nolock()
                return False

    def _disconnect_mq_nolock(self): # Version for use inside _mq_lock
        # ... (same as in RCI) ...
        pass
    def _disconnect_mq(self): # Public version
        # ... (same as in RCI) ...
        pass


    def _publish_command_to_mq(self, exchange_name, routing_key, command_body_dict, expect_reply=False, timeout_sec=10):
        """Publishes a command to MQ and optionally waits for a reply using correlation_id."""
        if not (self._mq_channel and self._mq_channel.is_open):
            logger.error(f"WPM: MQ channel not open for publishing command for {routing_key}.")
            if not self._connect_mq(): return None if expect_reply else False # Attempt reconnect
            # If still not connected after reconnect attempt, then fail
            if not (self._mq_channel and self._mq_channel.is_open):
                logger.error("WPM: MQ Reconnect failed. Command not sent.")
                return None if expect_reply else False


        correlation_id = str(uuid.uuid4()) if expect_reply else None
        reply_to_queue = self.response_queue_name if expect_reply else None

        try:
            message_body_str = json.dumps(command_body_dict, ensure_ascii=False)
            properties = pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
                content_type='application/json',
                timestamp=int(time.time()),
                correlation_id=correlation_id,
                reply_to=reply_to_queue
            )
            with self._mq_lock:
                self._mq_channel.basic_publish(exchange=exchange_name, routing_key=routing_key, body=message_body_str, properties=properties)
            logger.info(f"WPM: Published command to MQ. Exch='{exchange_name}', Key='{routing_key}', CorrID='{correlation_id}'")

            if not expect_reply:
                return True # Command sent, no reply awaited

            # Wait for reply
            response_event = threading.Event()
            response_data_holder = {"data": None} # Use a mutable object to hold response
            self._mq_response_events[correlation_id] = (response_event, response_data_holder)

            if response_event.wait(timeout=timeout_sec):
                # Event was set, response should be in response_data_holder
                del self._mq_response_events[correlation_id] # Clean up
                logger.info(f"WPM: Received reply for CorrID '{correlation_id}': {response_data_holder['data']}")
                return response_data_holder['data']
            else:
                # Timeout waiting for reply
                logger.warning(f"WPM: Timeout waiting for reply for CorrID '{correlation_id}' (Key: {routing_key}).")
                if correlation_id in self._mq_response_events: del self._mq_response_events[correlation_id]
                return None # Or a specific timeout error object

        except Exception as e:
            logger.error(f"WPM: Failed to publish command or process reply. Key: {routing_key}. Error: {e}", exc_info=True)
            if correlation_id and correlation_id in self._mq_response_events: del self._mq_response_events[correlation_id]
            if isinstance(e, (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker)): self._disconnect_mq()
            return None if expect_reply else False


    def _mq_response_consumer_loop(self):
        """Consumes messages from this WPM's dedicated reply queue."""
        logger.info("WPM MQ Response Consumer loop started.")
        while not self._stop_mq_event.is_set():
            try:
                if not (self._mq_channel and self._mq_channel.is_open):
                    logger.warning("WPM: MQ channel not open in response consumer loop. Reconnecting...")
                    if not self._connect_mq(): time.sleep(5); continue

                # This consumes from self.response_queue_name
                # This method is blocking, but process_data_events allows for periodic checks
                # It's simpler to use channel.consume for this callback-based approach
                # self._mq_channel.process_data_events(time_limit=1) # Check for messages with timeout
                # If using basic_consume, ensure it's started in a way that allows this loop to continue or use start_consuming
                # For BlockingConnection, start_consuming is typically used in its own thread.
                # So, this consumer loop should be where basic_consume is set up.

                # Let's set up basic_consume here for the reply queue
                self._mq_channel.basic_consume(queue=self.response_queue_name, on_message_callback=self._mq_reply_callback, auto_ack=True)
                logger.info(f"WPM: Starting to consume replies from MQ queue '{self.response_queue_name}'...")
                self._mq_channel.start_consuming() # This will block until stop_consuming or error
                logger.info("WPM: MQ reply consumption stopped.")
                if self._stop_mq_event.is_set(): break # Exit loop if stop was signaled


            except pika.exceptions.ConnectionClosedByBroker: logger.warning("WPM: MQ Connection (Reply) closed by broker. Reconnecting..."); self._disconnect_mq(); time.sleep(5)
            # Add other specific pika exceptions
            except Exception as e: logger.error(f"WPM: MQ Response Consumer loop error: {e}", exc_info=True); self._disconnect_mq(); time.sleep(5)
        logger.info("WPM MQ Response Consumer loop stopped.")


    def _mq_reply_callback(self, ch, method, properties, body):
        """Callback for processing reply messages received from MQ."""
        correlation_id = properties.correlation_id
        if correlation_id and correlation_id in self._mq_response_events:
            try:
                response_data = json.loads(body.decode('utf-8'))
                logger.debug(f"WPM MQ Reply Callback: Received reply for CorrID '{correlation_id}'.")
                event, data_holder = self._mq_response_events[correlation_id]
                data_holder["data"] = response_data # Store the response
                event.set() # Signal the waiting thread
                # No need to ack if auto_ack=True
            except json.JSONDecodeError:
                logger.error(f"WPM MQ Reply Callback: Failed to decode JSON from reply message: {body.decode('utf-8')[:200]}")
            except Exception as e:
                logger.error(f"WPM MQ Reply Callback: Error processing reply: {e}", exc_info=True)
                # If error, signal event anyway so waiter doesn't hang indefinitely, but data will be None
                event, data_holder = self._mq_response_events.get(correlation_id, (None, None))
                if event: data_holder["data"] = {"status":"error_processing_reply", "details": str(e)}; event.set()
        else:
            logger.warning(f"WPM MQ Reply Callback: Received reply with unknown/unawaited correlation_id: {correlation_id}. Body: {body.decode('utf-8')[:100]}")


    # --- Core Process Logic (Modified to use MQ for commands) ---
    def _execute_robot_action_mq(self, robot_id, action_details, timeout_sec=30):
        """Sends a robot action command via MQ and waits for response."""
        action_type = action_details.get("action")
        command_payload = {
            "robot_id": robot_id,
            "action": action_type,
            "parameters": action_details # Pass all details as parameters to RCI
        }
        # Routing key for robot commands (RCI specific queue or RCI exchange + robot_id key)
        # Assuming RCI consumes from a direct queue named in config.MQ_ROBOT_COMMAND_QUEUE
        robot_cmd_routing_key = getattr(config, 'MQ_ROBOT_COMMAND_QUEUE', f'robot_cmd_q_robot{robot_id}') # Or specific key
        robot_cmd_exchange = getattr(config, 'MQ_ROBOT_COMMAND_EXCHANGE', '') # Use default exchange if routing directly to queue

        logger.info(f"WPM: Sending Robot Action '{action_type}' to Robot {robot_id} via MQ.")
        # This _publish_command_to_mq will use self.response_queue_name as reply_to
        response = self._publish_command_to_mq(robot_cmd_exchange, robot_cmd_routing_key, command_payload, expect_reply=True, timeout_sec=timeout_sec)

        if response and response.get("status") == "success":
            logger.info(f"WPM: Robot Action '{action_type}' for Robot {robot_id} successful (via MQ). Response: {response.get('data') or response.get('message')}")
            # --- Visualization Update via direct call to physics_interface ---
            if self.physics_interface and self.physics_interface.is_connected:
                target_pose = action_details.get("target_pose")
                if action_type in ["move_to_start_pos", "move_to_home_pos"] and target_pose:
                    self.physics_interface.send_robot_pose(robot_id, joint_angles=target_pose.get('joint_angles'), tcp_transform=target_pose.get('tcp_transform'))
                elif action_type == "welder_on":
                    current_params = self._get_current_robot_params(robot_id) # Get params from internal state
                    self.physics_interface.send_welding_visual_command(robot_id, "arc_on", details={"welding_params": current_params})
                elif action_type == "welder_off":
                    self.physics_interface.send_welding_visual_command(robot_id, "arc_off")
            return True
        else:
            logger.error(f"WPM: Robot Action '{action_type}' for Robot {robot_id} failed or timed out (via MQ). Response: {response}")
            return False

    # _execute_group_action needs significant rewrite for MQ based async command execution and result aggregation.
    # For brevity, it's omitted in this MQ-focused example but would follow similar patterns.

    def _adaptive_control_loop(self, robot_id, welding_job_name, checkpoints):
        # ... (Similar to previous version, but AI predictions and Quality Evals now come from MQ subscriptions) ...
        # ... (Adjustments are published to MQ for RCI to consume) ...
        logger.info(f"WPM Robot {robot_id}: Starting ADVANCED adaptive control for job '{welding_job_name}'.")
        # This loop needs its own MQ subscription to AI Predictions and QEA Results for this robot_id/job_id.
        # Or, WPM's main MQ consumer loop can route these to the correct AC thread.
        # For simplicity here, assume WPM's main consumer routes data or this loop directly polls for it.

        # For real-time pose streaming to UE:
        # Use self.robot_interface_direct.get_latest_robot_pose(robot_id) for direct access
        # OR, subscribe to MQ_ROBOT_STATUS_EXCHANGE if RCI publishes pose there.
        # Direct access is simpler if RCI is in the same process or provides a quick API.
        pose_stream_interval = config.POSE_STREAM_INTERVAL_SEC

        while self._running and self.current_process_state in [WeldingProcessState.WELDING_IN_PROGRESS, WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE] and not self.stop_event.is_set():
            loop_start_time = time.time()
            # --- 1. Get AI Prediction (Concept: from an internal queue filled by MQ consumer) ---
            # ai_prediction = self._get_ai_prediction_for_robot(robot_id, timeout=0.1) # Needs implementation
            # For this example, let's simulate a direct request for simplicity, though MQ is preferred.
            # This would publish a request to AI Engine and wait for reply.
            # ai_request_payload = {"robot_id": robot_id, "sensor_data": self.sensor_handler.get_latest_aggregated_data(robot_id)} # Needs actual sensor data
            # ai_prediction_response = self._publish_command_to_mq(getattr(config,'MQ_AI_REQUEST_EXCHANGE',''), f"ai.request.robot{robot_id}", ai_request_payload, expect_reply=True, timeout_sec=1)
            # ai_prediction = ai_prediction_response.get('prediction') if ai_prediction_response and ai_prediction_response.get('status') == 'success' else None

            # --- For now, assume AI predictions and QEA results arrive via WPM's main MQ consumer and are routed ---
            # --- This AC loop would focus on *acting* on them and streaming pose ---

            # --- Real-time Robot Pose Streaming to UE ---
            if self.robot_interface_direct and self.physics_interface and self.physics_interface.is_connected:
                latest_pose = self.robot_interface_direct.get_latest_robot_pose(robot_id) # Direct call
                if latest_pose:
                    self.physics_interface.send_robot_pose(
                        robot_id=robot_id,
                        joint_angles=latest_pose.get('joint_angles'),
                        tcp_transform=latest_pose.get('tcp_transform')
                    )

            # --- Placeholder for receiving evaluated quality and applying adjustments ---
            # In a full MQ system, QEA would publish "adjustment_needed" messages.
            # This WPM AC loop would subscribe to those.
            # For now, we skip the detailed QEA interaction here, assuming it happens elsewhere
            # and commands for adjustment are sent directly to the robot command MQ by QEA or by WPM after QEA.

            # Control loop frequency for streaming
            elapsed_in_loop = time.time() - loop_start_time
            sleep_time = pose_stream_interval - elapsed_in_loop
            if sleep_time > 0: self.stop_event.wait(timeout=sleep_time) # Use stop_event for interruptible sleep
        logger.info(f"WPM Robot {robot_id}: Adaptive control / visualization stream for job '{welding_job_name}' ended.")


    # _process_job_thread_func, start_welding_job, stop_current_job, get_manager_status
    # need to be refactored to use the _execute_robot_action_mq and handle async MQ responses.
    # This is a significant change. The example below shows a conceptual start.

    def _process_job_thread_func(self, job_id):
        # ... (Initial setup, recipe loading, robot connection checks as before) ...
        # ... (Ensure physics_interface is checked for connection too) ...
        logger.info(f"WPM Job Thread {job_id}: Starting execution using MQ for robot commands.")
        # In this MQ-based version, _execute_robot_action_mq is used.
        # The loop needs to handle asynchronous nature of MQ commands/responses.
        # This might involve more complex state management (e.g., WAITING_ROBOT_RESPONSE).
        # For simplicity, the example shows sequential execution with blocking waits for MQ responses.
        # A fully async WPM would use callbacks or an event loop with the MQ consumer.
        # ...
        # Example for a step:
        # if self._execute_robot_action_mq(robot_id, step_details, timeout_sec=step_timeout):
        #     # Action successful
        # else:
        #     # Action failed or timed out
        #     self._set_process_state(WeldingProcessState.ERROR)
        #     break
        # ...
        # Start AC loops as before
        # ...
        # Wait for AC loops to finish
        # ...
        # Finalize job status
        # ...
        logger.warning("_process_job_thread_func for MQ is not fully implemented in this example.")
        self._set_process_state(WeldingProcessState.IDLE) # Ensure reset
        pass # Placeholder for full MQ-based job execution logic


    # --- Public Interface (called by SystemManager or HMI via SystemManager) ---
    def start(self):
        """Starts the WPM's MQ consumer thread."""
        if not config.USE_MESSAGE_QUEUE or not pika:
             logger.warning("WPM: MQ not enabled or Pika not available. WPM will not consume MQ messages.")
             return

        if self._mq_consumer_thread and self._mq_consumer_thread.is_alive():
            logger.info("WPM MQ consumer already running.")
            return

        if not (self._mq_channel and self._mq_channel.is_open):
            if not self._connect_mq():
                logger.error("WPM: Cannot start MQ consumer, failed to connect to MQ.")
                return

        logger.info("Starting WPM MQ consumer thread (for replies and other relevant topics)...")
        self._stop_mq_event.clear()
        self._mq_consumer_thread = threading.Thread(target=self._mq_response_consumer_loop, name="WPMMQConsumer", daemon=True)
        self._mq_consumer_thread.start()


    def stop(self):
        """Stops the WPM's MQ consumer thread and signals job processing to stop."""
        logger.info("WPM: Stopping...")
        self.stop_event.set() # Signal main job processing thread and AC loops to stop

        if self._mq_consumer_thread and self._mq_consumer_thread.is_alive():
            if self._mq_channel and self._mq_channel.is_open:
                try: self._mq_channel.stop_consuming() # Request consumer to stop
                except Exception as e: logger.error(f"WPM: Error stopping MQ consumer channel: {e}")
            logger.info("Waiting for WPM MQ consumer thread to join...")
            self._mq_consumer_thread.join(timeout=5.0)
            if self._mq_consumer_thread.is_alive(): logger.warning("WPM MQ consumer thread did not join.")
        self._mq_consumer_thread = None

        # Wait for main process_thread if it's running
        if self.process_thread and self.process_thread.is_alive():
             logger.info("Waiting for WPM main job processing thread to complete...")
             self.process_thread.join(timeout=10) # Give current job time to abort
             if self.process_thread.is_alive(): logger.warning("WPM main job processing thread did not join.")
        self.process_thread = None


        self._disconnect_mq()
        logger.info("WPM: Stopped.")

    # start_welding_job, stop_current_job, get_manager_status methods would be similar
    # to previous versions, but start_welding_job might trigger an initial MQ command
    # or just set up the _process_job_thread_func.
    # stop_current_job would use MQ to send stop commands if robots are controlled that way,
    # in addition to setting self.stop_event.


# Example (WPM is usually managed by SystemManager)
if __name__ == '__main__':
    logger.warning("WeldingProcessManager (Advanced with MQ) is designed to be orchestrated by SystemManager.")
    logger.info("Direct run example for WPM with MQ is highly complex and depends on other MQ publishers/consumers.")
    # To test:
    # 1. Ensure RabbitMQ is running.
    # 2. Ensure config.py is correctly set up for MQ and robot_configs.json, welding_recipes.json exist.
    # 3. Run a dummy RobotControlInterface that *consumes* commands from MQ_ROBOT_COMMAND_QUEUE
    #    and *publishes* responses to MQ_ROBOT_RESPONSE_EXCHANGE (with correlation_id and reply_to).
    # 4. Instantiate WPM and call start() to start its MQ consumer.
    # 5. Manually publish a "start_job" command (or simulate HMI call to SystemManager that does it)
    #    to an MQ queue that WPM's SystemManager integration part would listen to.
    #    This WPM example itself doesn't consume "start_job" from MQ yet; start_welding_job is direct call.