# src/robot_control_interface.py
# Description: Defines an interface to communicate with Hyundai Robotics controllers (Placeholder).
# Assumes TCP/IP socket communication with a specific (virtual) protocol.
# Manages multiple robot connections and provides real-time status/pose feedback.

import socket
import json
import time
import threading
import queue # For managing command queues or received data
import logging # Use standard logging

import config # Requires ROBOT_CONFIGS in config.py

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- Virtual Robot Communication Protocol (Placeholder) ---
# Commands from Python to Robot (JSON format over TCP):
# Request: { "action": "command_name", "parameters": { ... }, "sequence_id": 123 }
# Actions: "get_status", "run_job", "stop_job", "pause_job", "resume_job",
#          "send_path_data", "set_io", "get_io", "set_param", "get_param"

# Responses from Robot to Python (JSON format over TCP):
# Response: { "status": "success" | "failure", "sequence_id": 123,
#             "message": "...", "data": { ... } | "error": "..." }
# Status/Pose Push Updates (Optional, Simulated):
# { "type": "status_update", "robot_id": 1, "timestamp": "ISO8601", "data": { "position_tcp": [...], "joint_angles": [...], ... } }

# Message Framing: 4-byte length prefix (big-endian) followed by JSON payload.

# --- Configuration for Monitoring ---
# Define in config.py:
# ROBOT_STATUS_MONITOR_INTERVAL_SEC = 0.05 # How frequently to poll status (if polling) or check for pushed data
# ROBOT_STATUS_RECEIVE_TIMEOUT_SEC = 0.1 # Timeout for receiving data during monitoring cycle

class RobotConnection:
    """Manages a single TCP connection to one robot."""
    def __init__(self, robot_config, data_logger=None):
        self.robot_id = robot_config.get('id')
        self.name = robot_config.get('name', f'Robot{self.robot_id}')
        self.host = robot_config.get('ip')
        self.port = robot_config.get('port')
        self.socket = None
        self.is_connected = False
        self.data_logger = data_logger # DataLoggerDB singleton instance if provided

        self._sequence_id_counter = 0 # To track requests/responses
        self._send_lock = threading.Lock() # Ensure only one thread sends at a time on this socket
        self._receive_lock = threading.Lock() # Ensure only one thread receives at a time on this socket

        # --- Real-time Status Monitoring ---
        self._latest_status_data = {} # Stores the latest received status/pose data
        self._status_lock = threading.Lock() # Protects _latest_status_data
        self._monitoring_thread = None
        self._monitor_running = False
        self._monitor_stop_event = threading.Event()

        logger.info(f"Robot Connection '{self.name}' ({self.host}:{self.port}) initialized.")
        if self.robot_id is None or self.host is None or self.port is None:
             logger.error(f"Robot configuration for '{self.name}' is incomplete: {robot_config}")
             raise ValueError(f"Incomplete robot configuration for {self.name}")


    def connect(self):
        """Establishes connection to the robot."""
        if self.is_connected and self.socket:
            logger.info(f"Robot '{self.name}': Already connected.")
            return True

        logger.info(f"Robot '{self.name}': Attempting to connect to {self.host}:{self.port}")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5) # Timeout for connection attempt
            self.socket.connect((self.host, self.port))
            # Set a non-blocking or very short timeout for the monitoring thread's recv calls
            # But allow longer timeout for synchronous send/receive commands.
            # We'll manage timeouts per operation or in the _receive_framed_json method.
            self.socket.settimeout(10) # Default timeout for blocking operations

            self.is_connected = True
            logger.info(f"Robot '{self.name}': Successfully connected.")
            # Log connection event
            if self.data_logger:
                 self.data_logger.log_robot_status(self.robot_id, {"event": "connected", "address": f"{self.host}:{self.port}"})
            return True
        except ConnectionRefusedError:
            logger.error(f"Robot '{self.name}': Connection refused. Is the robot controller/simulator running at {self.host}:{self.port}?")
            if self.data_logger:
                 self.data_logger.log_robot_status(self.robot_id, {"event": "connection_failed", "reason": "refused"})
            self._cleanup_connection()
            return False
        except socket.timeout:
             logger.error(f"Robot '{self.name}': Connection timed out attempting to reach {self.host}:{self.port}.")
             if self.data_logger:
                 self.data_logger.log_robot_status(self.robot_id, {"event": "connection_failed", "reason": "timeout"})
             self._cleanup_connection()
             return False
        except Exception as e:
            logger.error(f"Robot '{self.name}': Error connecting to {self.host}:{self.port}: {e}")
            if self.data_logger:
                 self.data_logger.log_robot_status(self.robot_id, {"event": "connection_failed", "reason": str(e)})
            self._cleanup_connection()
            return False

    def disconnect(self):
        """Closes the connection and stops monitoring."""
        self.stop_monitoring() # Ensure monitoring thread is stopped first

        if self.is_connected and self.socket:
            logger.info(f"Robot '{self.name}': Disconnecting.")
            if self.data_logger:
                 self.data_logger.log_robot_status(self.robot_id, {"event": "disconnecting"})
            try:
                # Attempt a graceful shutdown first
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception as e:
                logger.error(f"Robot '{self.name}': Error during socket shutdown/close: {e}")
            finally:
                 self._cleanup_connection()
                 logger.info(f"Robot '{self.name}': Disconnected.")
                 if self.data_logger:
                      self.data_logger.log_robot_status(self.robot_id, {"event": "disconnected"})
        else:
             logger.info(f"Robot '{self.name}': Not connected.")

    def _cleanup_connection(self):
         """Internal helper to clean up socket state."""
         self.socket = None
         self.is_connected = False
         logger.debug(f"Robot '{self.name}': Connection state cleaned up.")


    def _send_framed_json(self, data):
        """Sends JSON data with 4-byte length prefix."""
        if not self.is_connected or not self.socket:
            # logger.warning(f"Robot '{self.name}': Not connected, cannot send data.") # Too frequent potentially
            return False
        try:
            message = json.dumps(data).encode('utf-8')
            message_len = len(message)
            # Ensure thread-safe sending
            with self._send_lock:
                 # Use a specific timeout for sending if needed, or rely on default socket timeout
                 self.socket.sendall(message_len.to_bytes(4, 'big')) # Send message size (4 bytes, big endian)
                 self.socket.sendall(message) # Send the actual JSON message
            # logger.debug(f"Robot '{self.name}': Sent {message_len} bytes.") # Enable for verbose debugging
            return True
        except socket.timeout:
            logger.error(f"Robot '{self.name}': Socket timeout during send. Disconnecting.")
            self.disconnect() # Assume connection is broken
            return False
        except Exception as e:
            logger.error(f"Robot '{self.name}': Error sending data: {e}. Disconnecting.")
            self.disconnect()
            return False

    def _receive_exactly(self, num_bytes):
        """Helper to receive exactly num_bytes from the socket."""
        data = b''
        while len(data) < num_bytes:
            # Use a small timeout for chunks if receiving large data,
            # or rely on overall operation timeout set by caller.
            # Relying on overall operation timeout is simpler here.
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk: # Connection closed by peer
                raise ConnectionResetError(f"Robot '{self.name}': Connection closed by peer.")
            data += chunk
        return data

    def _receive_framed_json(self, timeout=None):
        """Receives JSON data with 4-byte length prefix."""
        if not self.is_connected or not self.socket:
            return None # Return None immediately if not connected

        # Use a lock for thread-safe receiving
        with self._receive_lock:
            original_timeout = self.socket.gettimeout()
            try:
                if timeout is not None:
                    self.socket.settimeout(timeout)

                # Read message size (blocking call, respects socket timeout)
                raw_msglen = self._receive_exactly(4) # Use helper to ensure 4 bytes are read
                msglen = int.from_bytes(raw_msglen, 'big')
                # logger.debug(f"Robot '{self.name}': Expecting {msglen} bytes.")

                # Read the actual message
                data = self._receive_exactly(msglen)

                # Decode and parse the JSON message
                response = json.loads(data.decode('utf-8'))
                # logger.debug(f"Robot '{self.name}': Received data.") # Enable for verbose debugging
                return response

            except (socket.timeout, ConnectionResetError) as e:
                # Specific handling for timeout and connection reset during receive
                is_timeout = isinstance(e, socket.timeout)
                level = logging.WARNING if is_timeout else logging.ERROR
                logger.log(level, f"Robot '{self.name}': Receive error: {type(e).__name__} - {e}")
                if not is_timeout: # Connection errors imply disconnect
                    self.disconnect() # Disconnect on connection reset
                return {"status": "receive_error", "error": str(e), "is_timeout": is_timeout} # Indicate error/timeout state

            except json.JSONDecodeError:
                logger.error(f"Robot '{self.name}': Could not decode JSON response.")
                # Protocol error, consider logging raw data for debugging
                return {"status": "receive_error", "error": "JSON decode error"} # Indicate error

            except Exception as e:
                logger.error(f"Robot '{self.name}': Unexpected error during receive: {e}. Disconnecting.")
                self.disconnect()
                return {"status": "receive_error", "error": str(e)} # Indicate error

            finally:
                 # Restore original socket timeout if it was changed
                 if self.is_connected and self.socket:
                     self.socket.settimeout(original_timeout)


    def _generate_sequence_id(self):
        """Generates a unique sequence ID for requests."""
        with self._send_lock: # Use send lock as it's incremented before sending
             self._sequence_id_counter += 1
             return self._sequence_id_counter

    def send_command_and_wait_response(self, action, parameters=None, timeout=10):
        """
        Sends a command request and waits for the corresponding response.
        Args:
            action (str): The command action (e.g., "run_job").
            parameters (dict, optional): Dictionary of parameters for the command.
            timeout (int): Timeout for the entire send+receive operation.
        Returns:
            dict or None: The response dictionary from the robot, or None if failed/timeout/protocol error.
        """
        if not self.is_connected:
            logger.error(f"Robot '{self.name}': Cannot send command '{action}', not connected.")
            return None

        seq_id = self._generate_sequence_id()
        request = {
            "action": action,
            "parameters": parameters if parameters is not None else {},
            "sequence_id": seq_id
        }

        logger.info(f"Robot '{self.name}': Sending command: '{action}' (Seq ID: {seq_id})")
        if not self._send_framed_json(request):
            # Error already logged and disconnect handled in _send_framed_json
            return None # Send failed

        # Wait for the response with the matching sequence ID
        # A robust implementation would receive messages asynchronously into a queue
        # and match sequence IDs. For simplicity here, we'll block and expect the next
        # message to be the response. This is NOT suitable for concurrent commands
        # expecting responses on the same connection.
        logger.debug(f"Robot '{self.name}': Waiting for response for Seq ID {seq_id} with timeout {timeout}s.")
        response = self._receive_framed_json(timeout=timeout) # Use timeout for receive

        if response is None:
            # Connection likely closed during receive attempt
            logger.error(f"Robot '{self.name}': Failed to receive response for Seq ID {seq_id}. Connection likely closed.")
            return None
        elif response.get('status') == 'receive_error' and response.get('is_timeout'):
            # It was a receive timeout, not necessarily a disconnect
            logger.warning(f"Robot '{self.name}': Timeout waiting for response for Seq ID {seq_id}.")
            # Return the timeout status received from _receive_framed_json
            return response
        elif response.get('sequence_id') != seq_id:
             logger.warning(f"Robot '{self.name}': Received response with mismatching Seq ID. Expected {seq_id}, got {response.get('sequence_id')}. Response: {response}")
             # Depending on protocol, might need to handle this unexpected message (e.g., queue it for another handler)
             # For simplicity, treat as failure to get the *correct* response
             # The unexpected response might still be valid, so log it or put it aside before returning None
             return {"status": "protocol_error", "error": "Mismatching sequence ID", "received_response": response}
        elif response.get('status') == 'success':
            logger.info(f"Robot '{self.name}': Command '{action}' (Seq ID: {seq_id}) successful.")
            return response # Return the full successful response dictionary
        elif response.get('status') == 'failure':
            logger.error(f"Robot '{self.name}': Command '{action}' (Seq ID: {seq_id}) failed on robot side. Response: {response.get('error', 'No details')}")
            return response # Return the full failure response dictionary
        else:
            logger.error(f"Robot '{self.name}': Received unexpected response status for Seq ID {seq_id}: {response}. Assuming failure.")
            return response # Return the unexpected response


    # --- Real-time Status Monitoring Implementation ---

    def start_monitoring(self):
        """Starts a background thread to receive status/pose updates from the robot."""
        if self._monitor_running:
            logger.warning(f"Robot '{self.name}': Status monitoring is already active.")
            return
        if not self.is_connected:
             logger.warning(f"Robot '{self.name}': Cannot start monitoring, not connected.")
             return

        logger.info(f"Robot '{self.name}': Starting status monitor thread.")
        self._monitor_running = True
        self._monitor_stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, name=f'{self.name}-Monitor', daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stops the background status monitoring thread."""
        if self._monitor_running:
            logger.info(f"Robot '{self.name}': Stopping status monitor thread...")
            self._monitor_running = False
            self._monitor_stop_event.set() # Signal the thread to stop
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                # Wait for the thread to finish. Use a timeout.
                wait_timeout = getattr(config, 'ROBOT_MONITOR_SHUTDOWN_TIMEOUT_SEC', 2.0)
                self._monitoring_thread.join(timeout=wait_timeout)
                if self._monitoring_thread.is_alive():
                    logger.warning(f"Robot '{self.name}': Status monitor thread did not terminate gracefully.")
                else:
                     logger.info(f"Robot '{self.name}': Status monitor thread stopped.")
            self._monitoring_thread = None
        else:
            # logger.debug(f"Robot '{self.name}': Status monitoring is not active.") # Too verbose if called often

    def _monitor_loop(self):
        """Background thread loop to continuously receive data from the robot."""
        logger.info(f"Robot '{self.name}': Monitor loop started.")
        # Set a very short timeout for receiving in this loop
        # This allows the loop to check the stop event frequently even if no data arrives.
        receive_timeout = getattr(config, 'ROBOT_STATUS_RECEIVE_TIMEOUT_SEC', 0.1) # Example: 100ms timeout

        # In a polling scenario, you would send a 'get_status' request here periodically
        # poll_interval = getattr(config, 'ROBOT_STATUS_MONITOR_INTERVAL_SEC', 0.05) # Example: 50ms polling

        while self._monitor_running and not self._monitor_stop_event.is_set():
            try:
                # --- Receiving Logic (Assumes Robot Pushes Data or we are polling) ---
                # If robot PUSHES data:
                # Try to receive the next message. Use a short timeout.
                # This call blocks until data is received, timeout, or error.
                # received_message = self._receive_framed_json(timeout=receive_timeout)

                # If we are POLLING status:
                # Send a get_status request and wait for response.
                # This is simpler but might be slower for real-time updates.
                # Let's simulate receiving pushed updates as it's better for real-time vis.
                # So, uncomment the _receive_framed_json call below and comment out polling logic.

                # --- Simulate Pushed Status Update ---
                # In a real scenario, the line below would be self._receive_framed_json(timeout=receive_timeout)
                # And the robot controller would be sending unsolicited status messages.
                # For the dummy server, we need it to SEND these unsolicited messages.
                # Our dummy server currently only sends responses to commands.
                # Let's adjust the dummy server concept: it receives commands AND periodically sends status pushes.

                # For THIS client monitor loop, we just try to receive with a timeout.
                # received_message = self._receive_framed_json(timeout=receive_timeout)

                # --- ALTERNATIVE: Simulate Receiving Data ---
                # Since the dummy server doesn't yet push, let's *simulate* receiving data here
                # for testing the rest of the monitor loop logic (parsing, storing).
                # In a real implementation, replace this simulation with self._receive_framed_json.
                time.sleep(getattr(config, 'ROBOT_STATUS_MONITOR_INTERVAL_SEC', 0.05)) # Simulate arrival frequency
                dummy_pose = {
                    'position_tcp': [random.uniform(400, 600), random.uniform(-200, 200), random.uniform(150, 300)],
                    'joint_angles': [random.uniform(-180, 180) for _ in range(6)], # 6 joints
                    'speed_override': 100,
                    'job_running': "WELD_JOB" if random.random() > 0.2 else "IDLE",
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                received_message = {
                    "type": "status_update", # Our virtual push message type
                    "robot_id": self.robot_id,
                    "timestamp": dummy_pose['timestamp'],
                    "data": dummy_pose
                }
                # --- End Simulate Receiving Data ---


                if received_message is None:
                    # Connection closed by peer, _receive_framed_json handled disconnect
                    logger.warning(f"Robot '{self.name}': Monitor loop received None, assuming disconnect.")
                    break # Exit loop

                elif received_message.get('status') == 'receive_error' and received_message.get('is_timeout'):
                    # Timeout, no data received within the timeout. Continue waiting.
                    # logger.debug(f"Robot '{self.name}': Monitor receive timeout.") # Too verbose
                    pass

                elif received_message.get('type') == 'status_update' and received_message.get('robot_id') == self.robot_id:
                    # This is a status update message from the robot (unsolicited push)
                    status_data = received_message.get('data', {})
                    timestamp_str = received_message.get('timestamp')

                    if status_data:
                        # --- Store the latest status data ---
                        with self._status_lock:
                            self._latest_status_data = status_data
                            if timestamp_str:
                                self._latest_status_data['timestamp_received'] = timestamp_str
                            else:
                                self._latest_status_data['timestamp_received'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                            self._latest_status_data['is_latest'] = True # Mark as latest received
                            # Invalidate previous latest if tracking history
                            # for key, item in self._status_history: if key != current_key: item['is_latest'] = False

                        # Log the status update (potentially less frequently than receiving)
                        # Decide if logging every single update is necessary or if sampling is better.
                        # For now, log every update received by the monitor.
                        if self.data_logger:
                             try:
                                self.data_logger.log_robot_status(
                                    robot_id=self.robot_id,
                                    status_data=self._latest_status_data, # Log the full structure
                                    # job_id=... # Need mechanism to get current job ID here if needed
                                )
                             except Exception as log_e:
                                  logger.error(f"Robot '{self.name}': Error logging robot status: {log_e}")

                        # logger.debug(f"Robot '{self.name}': Stored latest status.") # Too verbose

                else:
                    # Received a message, but it's not a status update or not for this robot
                    # It might be a response to a command sent from send_command_and_wait_response,
                    # which we are NOT expecting here (it blocks).
                    # This indicates a potential protocol issue or unexpected message.
                    logger.warning(f"Robot '{self.name}': Monitor received unexpected message type or ID: {received_message}")
                    # Decide how to handle: ignore, log as error, put in a separate queue for unexpected messages?


            except Exception as e:
                logger.error(f"Robot '{self.name}': Unexpected error in monitor loop: {e}", exc_info=True)
                # If a critical error occurs, potentially disconnect or try to re-establish connection
                self.disconnect() # Assuming error might indicate broken connection
                break # Exit loop on error


        logger.info(f"Robot '{self.name}': Monitor loop finished.")
        self._monitor_running = False # Ensure flag is False on exit


    def get_latest_status(self):
        """Retrieves the latest raw status data received by the monitor."""
        with self._status_lock:
            # Return a copy to prevent external modification of internal state
            return self._latest_status_data.copy()

    def get_latest_pose(self):
        """
        Extracts and returns the latest known pose (joint angles or TCP)
        from the status data received by the monitor.
        Returns None if no status data is available or pose is not in the data.
        """
        latest_status = self.get_latest_status()
        if not latest_status:
            # logger.debug(f"Robot '{self.name}': No latest status data for pose.") # Too verbose
            return None

        # Extract pose data based on expected keys in the status data structure
        # This needs to match the format the robot controller sends (or simulates sending).
        pose_data = {}
        if 'joint_angles' in latest_status:
            pose_data['joint_angles'] = latest_status['joint_angles']
        if 'position_tcp' in latest_status: # Assuming TCP position
             pose_data['position_tcp'] = latest_status['position_tcp']
        if 'orientation_tcp' in latest_status: # Assuming TCP orientation (format TBD, e.g., quaternion, Euler)
             pose_data['orientation_tcp'] = latest_status['orientation_tcp'] # Example key

        if pose_data:
            # logger.debug(f"Robot '{self.name}': Extracted latest pose.") # Too verbose
            return pose_data # Return extracted pose data
        else:
            # logger.debug(f"Robot '{self.name}': Pose data keys not found in latest status.") # Too verbose
            return None # Pose data not present in the latest status structure


class RobotControlInterface:
    """
    Manages connections and communication with multiple robot controllers.
    Provides access to real-time status and allows sending commands.
    """
    def __init__(self, data_logger=None):
        # Get DataLoggerDB singleton instance if not provided
        # DataLoggerDB is a singleton, but passing it ensures SystemManager manages its lifecycle
        self.data_logger = data_logger # Assume DataLoggerDB instance is passed
        # Or get singleton: self.data_logger = DataLoggerDB()

        # Load robot configurations from config.py
        # Assuming config.py has a list like:
        # ROBOT_CONFIGS = [{'id': 1, 'name': 'Robot1', 'ip': '192.168.0.11', 'port': 5000}, ...]
        self.robot_configs = getattr(config, 'ROBOT_CONFIGS', [])
        if not self.robot_configs:
            logger.warning("No robot configurations found in config.py (looking for ROBOT_CONFIGS). Robot interface will be non-functional.")

        # Dictionary to hold RobotConnection instances
        self.connections = {} # {robot_id: RobotConnection instance}
        self._initialized_connections = False

        # Create RobotConnection instances during initialization
        if self.robot_configs:
             logger.info(f"Initializing {len(self.robot_configs)} RobotConnection instances.")
             try:
                 for robot_cfg in self.robot_configs:
                     # Use get to avoid KeyError if keys are missing, validation is in RobotConnection __init__
                     r_id = robot_cfg.get('id')
                     if r_id is not None:
                         # Pass data_logger instance to each connection for direct logging
                         conn = RobotConnection(robot_cfg, data_logger=self.data_logger)
                         self.connections[r_id] = conn
                     else:
                         logger.error(f"Skipping robot configuration due to missing 'id': {robot_cfg}")
                 self._initialized_connections = True
             except Exception as e:
                 logger.critical(f"Error initializing RobotConnection instances: {e}", exc_info=True)
                 self.connections = {} # Clear connections on error
                 self._initialized_connections = False


        self.connected_robot_ids = [] # List of robot_ids that are currently connected

        logger.info("Robot Control Interface initialized.")


    def connect_all(self):
        """Attempts to connect to all configured robots concurrently."""
        if not self._initialized_connections:
             logger.warning("Robot connections not initialized. Cannot connect.")
             return []

        logger.info("Attempting to connect to all configured robots...")
        self.connected_robot_ids = [] # Reset list
        threads = []
        results_queue = queue.Queue() # Queue to collect results from connection threads

        def _connect_single_robot_task(conn):
             success = conn.connect()
             results_queue.put({'id': conn.robot_id, 'connected': success, 'instance': conn})

        for robot_id, conn in self.connections.items():
            # Use threading to attempt connections concurrently
            thread = threading.Thread(target=_connect_single_robot_task, args=(conn,))
            threads.append(thread)
            thread.start()

        # Wait for all connection attempts to finish
        for thread in threads:
            thread.join()

        # Process results from the queue
        while not results_queue.empty():
             result = results_queue.get()
             if result['connected']:
                 self.connected_robot_ids.append(result['id'])
             # The RobotConnection instance is already stored in self.connections

        logger.info(f"Connection process finished. Successfully connected to robots: {self.connected_robot_ids}")
        return self.connected_robot_ids # Return list of IDs that connected

    def disconnect_all(self):
        """Disconnects from all connected robots."""
        if not self._initialized_connections:
             logger.warning("Robot connections not initialized. No connections to disconnect.")
             return

        logger.info("Disconnecting from all connected robots...")
        threads = []
        # Iterate over a copy of connections dictionary keys as disconnect might modify it
        for robot_id in list(self.connections.keys()):
             conn = self.connections.get(robot_id)
             if conn: # Ensure conn exists (might have failed init)
                 # Use threading to disconnect concurrently
                 thread = threading.Thread(target=conn.disconnect)
                 threads.append(thread)
                 thread.start()
             else:
                 logger.warning(f"Skipping disconnect for robot ID {robot_id}, connection instance not found.")


        # Wait for all disconnection attempts to finish
        for thread in threads:
            thread.join()

        self.connected_robot_ids = [] # Clear list of connected IDs
        logger.info("Disconnected from all robots.")

    def _get_connection(self, robot_id):
        """Gets the connection object for a specific robot, checking connectivity."""
        conn = self.connections.get(robot_id)
        if conn is None:
            logger.error(f"Robot {robot_id}: Connection instance not found.")
            return None
        if not conn.is_connected:
            logger.error(f"Robot {robot_id}: Not connected.")
            # Optionally try to reconnect here, but better to manage connection
            # status and reconnection attempts in SystemManager or a dedicated component.
            return None
        return conn

    # --- Public Methods for Robot Interaction (Wrapper around send_command_and_wait_response) ---
    # These methods call the lower-level communication and parse the response status.

    def get_robot_status(self, robot_id, timeout=5):
        """Requests and returns the status data of a specific robot."""
        conn = self._get_connection(robot_id)
        if conn:
            # Use send_command_and_wait_response which handles seq_id, send/receive.
            response = conn.send_command_and_wait_response("get_status", timeout=timeout)
            if response and response.get('status') == 'success':
                # logger.debug(f"Robot {robot_id}: Status received.") # Too verbose
                # Assuming status data is in the 'data' field of a successful response
                status_data = response.get('data')
                # Optionally update internal latest status here as well,
                # although the monitoring thread is the primary source.
                # with conn._status_lock: conn._latest_status_data = status_data # Careful with timestamp/full data
                # Log the response received for command
                if self.data_logger:
                    try:
                        self.data_logger.log_robot_status(robot_id, {"event": "get_status_response", "data": status_data}, job_id=self.current_job_id) # Needs job_id from WPM? Or log in WPM
                    except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging get_status response: {log_e}")

                return status_data
            elif response:
                 # Log the failed response
                 if self.data_logger:
                     try:
                          self.data_logger.log_robot_status(robot_id, {"event": "get_status_failed_response", "response": response}, job_id=self.current_job_id)
                     except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging failed get_status response: {log_e}")
                 # Log error details based on response status/error fields
                 if response.get('status') == 'receive_error':
                     logger.error(f"Robot {robot_id}: Failed to receive complete status response. Error: {response.get('error')}")
                 elif response.get('status') == 'timeout':
                     logger.warning(f"Robot {robot_id}: Timeout waiting for status response.")
                 else:
                     logger.error(f"Robot {robot_id}: Get status command failed on robot side. Response: {response}")
            # Return None or a dictionary indicating failure state
            return None # Or {'status': 'failed', 'details': response}


    def run_job(self, robot_id, job_name, parameters=None, timeout=20):
        """Sends command to run a specific job."""
        conn = self._get_connection(robot_id)
        if conn:
            # Use send_command_and_wait_response
            response = conn.send_command_and_wait_response("run_job", parameters={'job_name': job_name, 'params': parameters}, timeout=timeout)
            if response and response.get('status') == 'success':
                logger.info(f"Robot {robot_id}: Job '{job_name}' started command successful.")
                # Log event
                if self.data_logger:
                     try:
                          self.data_logger.log_robot_status(robot_id, {"event": "command_sent", "command": "run_job", "job_name": job_name, "status": "success_response"})
                     except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging run_job success: {log_e}")
                return True # Command acknowledged as successful
            elif response:
                 logger.error(f"Robot {robot_id}: Failed to run job '{job_name}' command. Response: {response}")
                 if self.data_logger:
                      try:
                           self.data_logger.log_robot_status(robot_id, {"event": "command_failed", "command": "run_job", "job_name": job_name, "response": response})
                      except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging run_job failure: {log_e}")
            # Return False on failure (including timeout or receive error)
            return False


    def stop_job(self, robot_id, timeout=10):
        """Sends command to stop the current job."""
        conn = self._get_connection(robot_id)
        if conn:
            response = conn.send_command_and_wait_response("stop_job", timeout=timeout)
            if response and response.get('status') == 'success':
                logger.info(f"Robot {robot_id}: Stop job command successful.")
                 # Log event
                if self.data_logger:
                     try:
                          self.data_logger.log_robot_status(robot_id, {"event": "command_sent", "command": "stop_job", "status": "success_response"})
                     except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging stop_job success: {log_e}")
                return True # Command acknowledged as successful
            elif response:
                 logger.error(f"Robot {robot_id}: Failed to stop job command. Response: {response}")
                 if self.data_logger:
                      try:
                           self.data_logger.log_robot_status(robot_id, {"event": "command_failed", "command": "stop_job", "response": response})
                      except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging stop_job failure: {log_e}")
            return False # Return False on failure

    # Add methods for pause_job, resume_job similarly...

    def set_welder_io(self, robot_id, io_value: bool, timeout=5):
        """Sets the state of the welder digital output (e.g., True/False)."""
        conn = self._get_connection(robot_id)
        if conn:
            # Assuming a specific IO point maps to the welder
            # 'welder_output' is a virtual IO name for this example
            response = conn.send_command_and_wait_response("set_io", parameters={'io_name': 'welder_output', 'value': io_value}, timeout=timeout)
            if response and response.get('status') == 'success':
                logger.info(f"Robot {robot_id}: Welder IO set to {io_value} command successful.")
                 # Log event
                if self.data_logger:
                     try:
                          self.data_logger.log_robot_status(robot_id, {"event": "command_sent", "command": "set_io", "io_name": "welder_output", "value": io_value, "status": "success_response"})
                     except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging set_io success: {log_e}")
                return True # Command acknowledged as successful
            elif response:
                 logger.error(f"Robot {robot_id}: Failed to set welder IO to {io_value} command. Response: {response}")
                 if self.data_logger:
                      try:
                           self.data_logger.log_robot_status(robot_id, {"event": "command_failed", "command": "set_io", "io_name": "welder_output", "value": io_value, "response": response})
                      except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging set_io failure: {log_e}")
            return False # Return False on failure

    def set_welding_parameters(self, robot_id, current=None, voltage=None, speed=None, timeout=5):
        """
        Sends commands to set welding parameters (current, voltage, speed).
        This might be done via setting specific robot registers or calling a dedicated function.
        """
        conn = self._get_connection(robot_id)
        if conn:
            # Filter None parameters to avoid sending them if not specified
            params_to_set = {}
            if current is not None: params_to_set['welding_current'] = current
            if voltage is not None: params_to_set['welding_voltage'] = voltage
            if speed is not None: params_to_set['welding_speed'] = speed

            if not params_to_set:
                logger.warning(f"Robot {robot_id}: set_welding_parameters called with no parameters to set.")
                return True # Nothing to do, count as success

            # This assumes the robot protocol has a single command to set multiple params,
            # or that we send multiple set_param commands and handle overall success.
            # Let's assume a single command 'set_welding_params' for simplicity in WPM interaction.
            # If robot requires multiple 'set_param' calls, this method needs internal logic to do that.
            # Assuming one command for now:
            response = conn.send_command_and_wait_response("set_welding_parameters", parameters=params_to_set, timeout=timeout)
            if response and response.get('status') == 'success':
                logger.info(f"Robot {robot_id}: Set welding parameters command successful.")
                # Log event
                if self.data_logger:
                     try:
                          self.data_logger.log_robot_status(robot_id, {"event": "command_sent", "command": "set_welding_parameters", "params": params_to_set, "status": "success_response"})
                     except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging set_welding_parameters success: {log_e}")
                return True # Command acknowledged as successful
            elif response:
                 logger.error(f"Robot {robot_id}: Failed to set welding parameters command. Response: {response}")
                 if self.data_logger:
                      try:
                           self.data_logger.log_robot_status(robot_id, {"event": "command_failed", "command": "set_welding_parameters", "params": params_to_set, "response": response})
                      except Exception as log_e: logger.error(f"Robot {robot_id}: Error logging set_welding_parameters failure: {log_e}")
            return False # Return False on failure


    # --- Methods for Real-time Status/Pose Access ---

    def start_monitoring_all(self):
        """Starts status monitoring threads for all connected robots."""
        logger.info("Starting status monitoring for all connected robots...")
        for robot_id in self.connected_robot_ids:
             conn = self.connections.get(robot_id)
             if conn and conn.is_connected:
                  conn.start_monitoring()
             else:
                  logger.warning(f"Cannot start monitoring for Robot {robot_id}, not connected.")

    def stop_monitoring_all(self):
        """Stops status monitoring threads for all robots."""
        logger.info("Stopping status monitoring for all robots...")
        # Iterate over all known connections, not just currently connected,
        # as monitoring thread might still exist but connection dropped.
        for robot_id, conn in self.connections.items():
             if conn: # Ensure conn instance exists
                 conn.stop_monitoring()

    def get_latest_robot_status(self, robot_id):
        """Retrieves the latest raw status data received by the monitor for a specific robot."""
        conn = self.connections.get(robot_id)
        if conn:
             # Returns a copy of the latest status or empty dict
             return conn.get_latest_status()
        else:
             logger.warning(f"Robot {robot_id}: Connection instance not found for getting latest status.")
             return {} # Return empty dict if robot not found

    def get_latest_robot_pose(self, robot_id):
        """
        Retrieves the latest known pose (joint angles or TCP) from the monitor thread
        for a specific robot.
        Returns None if robot not found, not connected, no status yet, or pose data missing.
        """
        conn = self.connections.get(robot_id)
        if conn:
            # Returns extracted pose data or None
             return conn.get_latest_pose()
        else:
             logger.warning(f"Robot {robot_id}: Connection instance not found for getting latest pose.")
             return None # Return None if robot not found


    # --- Other Utility Methods ---
    def is_robot_connected(self, robot_id):
        """Checks if a specific robot is currently connected."""
        conn = self.connections.get(robot_id)
        return conn is not None and conn.is_connected

    def get_connected_robot_ids(self):
        """Returns a list of IDs for robots that are currently connected."""
        # The self.connected_robot_ids list is updated by connect_all/disconnect_all
        # For dynamic connection status, you might need to iterate self.connections
        # and check conn.is_connected for each, but connect_all/disconnect_all is the primary way connection status is managed here.
        # Let's ensure self.connected_robot_ids is accurate based on current connections:
        self.connected_robot_ids = [r_id for r_id, conn in self.connections.items() if conn.is_connected]
        return self.connected_robot_ids


# Example Usage (requires dummy robot server(s))
# This __main__ block is for testing this specific module in isolation.
# The __main__ block in system_manager.py provides the example run for the whole system.
if __name__ == '__main__':
    # Set logging level higher for less verbose output during example run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Use the module-level logger created at the top
    logger.info("--- Robot Control Interface Example ---")

    # --- Dummy Config (for example purposes only) ---
    class DummyConfig:
        ROBOT_CONFIGS = [
            {'id': 1, 'name': 'Robot1', 'ip': '127.0.0.1', 'port': 6001},
            {'id': 2, 'name': 'Robot2', 'ip': '127.0.0.1', 'port': 6002}, # Second dummy robot
        ]
        # Config for monitoring thread timeouts
        ROBOT_STATUS_MONITOR_INTERVAL_SEC = 0.05 # Simulate updates every 50ms
        ROBOT_STATUS_RECEIVE_TIMEOUT_SEC = 0.1 # Short timeout for monitor recv
        ROBOT_MONITOR_SHUTDOWN_TIMEOUT_SEC = 0.5 # Quick shutdown timeout for monitor
        # Need a dummy DataLoggerDB for RobotConnection and Interface to log
        DATABASE_PATH = 'test_robot_interface_log.db' # Example DB path

    config = DummyConfig()

    # --- Dummy DataLoggerDB (for example purposes) ---
    # This is needed because the RobotConnection and RobotControlInterface try to use it.
    class DummyDataLoggerDB:
        # Singleton pattern placeholder - actual DataLoggerDB is a singleton
        _instance = None
        _lock = threading.Lock()
        def __new__(cls, *args, **kwargs):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.conn = True # Simulate a connection
                return cls._instance
        def log_robot_status(self, robot_id, status_data, job_id=None):
            # Print status data directly in the example
            print(f"LOG (Robot {robot_id}): {status_data}")
        def log_process_event(self, event_type, job_id=None, details=None):
             print(f"LOG (Process Event): {event_type} | Job: {job_id} | Details: {details}")
        def close_connection(self):
             print("LOG (DB): Connection closed (dummy).")

    dummy_logger = DummyDataLoggerDB()


    # --- Dummy Robot Server(s) (Placeholder for actual robot controller) ---
    # These servers simulate robot controllers listening for commands AND pushing status updates.
    def dummy_robot_server(robot_id, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1)
            logger.info(f"[DummyRobot {robot_id}] Listening on {host}:{port}")

            conn, addr = server_socket.accept()
            logger.info(f"[DummyRobot {robot_id}] Accepted connection from {addr}")
            # Set non-blocking or very short timeout for client connection
            # so the server can also send push updates.
            conn.settimeout(0.1) # Short timeout for recv

            # --- Server Push Status Thread ---
            def _push_status_loop():
                logger.info(f"[DummyRobot {robot_id}] Push status loop started.")
                while True: # Loop until connection is broken or signaled
                    try:
                        # Simulate generating status data (including pose)
                        current_pose = {
                             'position_tcp': [random.uniform(400, 600), random.uniform(-200, 200), random.uniform(150, 300)],
                             'joint_angles': [random.uniform(-180, 180) for _ in range(6)],
                             'speed_override': random.randint(50, 100),
                             'job_running': "WELD_JOB" if random.random() > 0.2 else "IDLE",
                             'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        push_message = {
                             "type": "status_update",
                             "robot_id": robot_id,
                             "timestamp": current_pose['timestamp'],
                             "data": current_pose
                        }
                        # Send the push message (use a lock if conn socket is used by main server logic too)
                        message = json.dumps(push_message).encode('utf-8')
                        conn.sendall(len(message).to_bytes(4, 'big'))
                        conn.sendall(message)
                        # logger.debug(f"[DummyRobot {robot_id}] Pushed status update.") # Too verbose

                        time.sleep(getattr(config, 'ROBOT_STATUS_MONITOR_INTERVAL_SEC', 0.05)) # Push frequency

                    except Exception as push_e:
                        # Connection broken, client likely disconnected
                        logger.warning(f"[DummyRobot {robot_id}] Error pushing status: {push_e}. Exiting push loop.")
                        break # Exit thread loop
                logger.info(f"[DummyRobot {robot_id}] Push status loop finished.")

            push_thread = threading.Thread(target=_push_status_loop, daemon=True)
            push_thread.start()


            # --- Server Command Receive Loop ---
            while True: # Handle multiple commands on one connection
                try:
                    # Receive message size and data
                    raw_msglen = conn.recv(4)
                    if not raw_msglen: break # Client disconnected
                    msglen = int.from_bytes(raw_msglen, 'big')
                    data = conn.recv(msglen) # Assuming msglen fits in one recv for simplicity
                    # In a real server, use a loop to receive exactly msglen bytes
                    if len(data) < msglen: break # Did not receive full message

                    request = json.loads(data.decode('utf-8'))
                    logger.info(f"[DummyRobot {robot_id}] Received request: {request.get('action')} (Seq ID: {request.get('sequence_id')})")

                    # Process request and prepare response (simplified)
                    response_data = {"status": "success", "sequence_id": request.get("sequence_id"), "message": "Acknowledged"}
                    action = request.get('action')

                    if action == "get_status":
                        # Respond with latest status (this is for the explicit get_status command, not the push)
                        response_data["data"] = dummy_pose # Use the last simulated pose for response
                    elif action == "run_job":
                        logger.info(f"[DummyRobot {robot_id}] Simulating job '{request.get('parameters', {}).get('job_name')}' for 3s...")
                        # time.sleep(3) # Don't block the server, simulate background job running
                        response_data["message"] = f"Job {request.get('parameters', {}).get('job_name')} start command acknowledged."
                    elif action == "stop_job":
                         logger.info(f"[DummyRobot {robot_id}] Simulating stopping job.")
                         response_data["message"] = "Stop command acknowledged."
                    elif action == "set_welding_parameters":
                         logger.info(f"[DummyRobot {robot_id}] Simulating setting params: {request.get('parameters')}")
                         response_data["message"] = "Set params command acknowledged."
                    elif action == "set_io":
                         logger.info(f"[DummyRobot {robot_id}] Simulating setting IO: {request.get('parameters')}")
                         response_data["message"] = "Set IO command acknowledged."

                    # Send response
                    response_message = json.dumps(response_data).encode('utf-8')
                    conn.sendall(len(response_message).to_bytes(4, 'big'))
                    conn.sendall(response_message)
                    logger.debug(f"[DummyRobot {robot_id}] Sent response for Seq ID {request.get('sequence_id')}")

                except socket.timeout:
                     # Timeout means no command received within the timeout, which is fine
                     # if the client isn't sending commands but only receiving pushes.
                     pass # Keep listening
                except ConnectionResetError:
                     logger.warning(f"[DummyRobot {robot_id}] Connection reset by client.")
                     break # Exit inner loop
                except Exception as e_inner:
                    logger.error(f"[DummyRobot {robot_id}] Error handling client command: {e_inner}", exc_info=True)
                    # Send an error response if possible before breaking
                    try:
                         err_response = {"status": "failure", "error": f"Server internal error: {e_inner}", "sequence_id": request.get("sequence_id")}
                         err_message = json.dumps(err_response).encode('utf-8')
                         conn.sendall(len(err_message).to_bytes(4, 'big'))
                         conn.sendall(err_message)
                    except: pass # Ignore errors sending error response
                    break # Exit inner loop

        except Exception as e_outer:
            logger.error(f"[DummyRobot {robot_id}] Server error: {e_outer}", exc_info=True)
        finally:
            logger.info(f"[DummyRobot {robot_id}] Client disconnected. Server waiting for new connection.")
            # conn socket is closed by 'with conn' or break.
            # push_thread might still be alive, needs a way to signal it to stop when conn breaks.
            # A dedicated connection handler class in the server would manage this.
            pass # Server socket remains open to accept new connections


    # Start dummy server(s) in separate threads for testing
    server_threads = []
    for cfg in config.ROBOT_CONFIGS:
        st = threading.Thread(target=dummy_robot_server, args=(cfg['id'], cfg['ip'], cfg['port']), name=f'DummyServer-{cfg["id"]}', daemon=True)
        server_threads.append(st)
        st.start()
        time.sleep(0.1) # Give server a moment to start

    if not server_threads:
        logger.error("No dummy servers started. Example will likely fail to connect.")


    # --- Initialize and Use the Interface ---
    # Need a dummy DataLoggerDB instance for the interface
    dummy_logger = DummyDataLoggerDB()
    robot_interface = RobotControlInterface(data_logger=dummy_logger)

    # Test Connection
    if robot_interface.connect_all():
        logger.info(f"\nSuccessfully connected to robots: {robot_interface.get_connected_robot_ids()}")

        if robot_interface.get_connected_robot_ids():
            test_robot_id = robot_interface.get_connected_robot_ids()[0] # Use the first connected robot

            # Start Status Monitoring for the test robot
            logger.info(f"\n--- Starting Status Monitoring for Robot {test_robot_id} ---")
            conn = robot_interface._get_connection(test_robot_id) # Get the connection instance
            if conn:
                 conn.start_monitoring()
                 logger.info("Monitoring started. Waiting for initial status updates...")
                 time.sleep(0.5) # Wait to receive a few dummy status updates

                 # Test Get Latest Status/Pose (polling from the stored data)
                 logger.info(f"\n--- Testing Get Latest Status/Pose from Robot {test_robot_id} ---")
                 latest_status = robot_interface.get_latest_robot_status(test_robot_id)
                 latest_pose = robot_interface.get_latest_robot_pose(test_robot_id)
                 if latest_status:
                     logger.info(f"Robot {test_robot_id} Latest Full Status: {latest_status}")
                 else:
                     logger.warning(f"Robot {test_robot_id} No latest status data available.")

                 if latest_pose:
                     logger.info(f"Robot {test_robot_id} Latest Pose: {latest_pose}")
                 else:
                      logger.warning(f"Robot {test_robot_id} No latest pose data available.")


                 # Test sending commands (these block waiting for response)
                 logger.info(f"\n--- Testing Sending Commands to Robot {test_robot_id} ---")
                 # Test Run Job
                 if robot_interface.run_job(test_robot_id, "WELD_PROGRAM_001"):
                     logger.info("Run job command sent and acknowledged.")

                 # Test Set Welding Parameters
                 if robot_interface.set_welding_parameters(test_robot_id, current=155, voltage=22.5):
                      logger.info("Set welding parameters command sent and acknowledged.")

                 # Test Set Welder IO
                 if robot_interface.set_welder_io(test_robot_id, True):
                     logger.info("Set welder IO ON command sent and acknowledged.")
                     time.sleep(0.5) # Simulate arc being on
                     if robot_interface.set_welder_io(test_robot_id, False):
                          logger.info("Set welder IO OFF command sent and acknowledged.")
                 else:
                      logger.warning("Failed to set welder IO ON.")


                 # Test Stop Job (if a job was simulated as running)
                 # Note: Dummy server simulation doesn't handle job lifecycle internally based on commands.
                 # This stop_job call will just get an acknowledgment from the dummy server.
                 # logger.info(f"\n--- Testing Stop Job for Robot {test_robot_id} ---")
                 # if robot_interface.stop_job(test_robot_id):
                 #      logger.info("Stop job command sent and acknowledged.")

                 # Wait for a few more status updates while commands are processed/simulated
                 logger.info("\nWaiting for a few more status updates...")
                 time.sleep(1.0) # Let monitor thread run

                 # Stop Status Monitoring
                 logger.info(f"\n--- Stopping Status Monitoring for Robot {test_robot_id} ---")
                 conn.stop_monitoring()


        # Disconnect from all robots at the end
        logger.info("\n--- Disconnecting from all robots ---")
        robot_interface.disconnect_all()
    else:
        logger.error("Failed to connect to any robots in the example.")

    # The dummy server threads are daemons and will exit when the main thread exits.
    # Clean up dummy database file if used
    # import os
    # if hasattr(config, 'DATABASE_PATH') and os.path.exists(config.DATABASE_PATH):
    #      logger.info(f"Removing test database: {config.DATABASE_PATH}")
    #      os.remove(config.DATABASE_PATH)


    logger.info("--- Robot Control Interface Example Finished ---")