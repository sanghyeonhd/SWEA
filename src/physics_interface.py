# src/physics_interface.py
# Description: (MODIFIED) Defines an interface to communicate with the Unreal Engine
#              physics simulator/visualizer. Supports basic async concepts,
#              batch simulation requests, and enhanced connection management.

import json
import socket
import time
import logging
import threading
import queue # For response queue
import uuid # For generating unique request IDs

from src import config # Import the main config module

# Setup logging for this module
logger = logging.getLogger(__name__)

class UnrealSimulatorInterface:
    """
    Manages communication with the Unreal Engine simulator/visualizer.
    Handles sending requests and processing responses, with considerations for
    asynchronous operations and batch requests.
    """
    _instance = None # Singleton pattern
    _lock = threading.Lock()

    # Connection & Reconnection Parameters
    RECONNECT_INTERVAL_SEC = getattr(config, 'UE_RECONNECT_INTERVAL_SEC', 5)
    MAX_RECONNECT_ATTEMPTS = getattr(config, 'UE_MAX_RECONNECT_ATTEMPTS', 3) # 0 for indefinite

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, host=None, port=None):
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            self.host = host if host is not None else config.UE_SIMULATOR_IP
            self.port = port if port is not None else config.UE_SIMULATOR_PORT
            self.socket = None
            self.is_connected = False
            self._socket_lock = threading.Lock() # Protects socket operations

            # For handling asynchronous responses and pushed data
            self._response_queue = queue.Queue() # Stores (request_id, response_data) or pushed_data
            self._response_wait_events = {} # {request_id: threading.Event()}
            self._push_data_callbacks = [] # List of functions to call when unsolicited data arrives
            self._receive_thread = None
            self._stop_receive_event = threading.Event()

            self._connection_thread = None # For handling initial connection and reconnections
            self._stop_connection_event = threading.Event()

            logger.info(f"Unreal Simulator Interface initialized for {self.host}:{self.port}")
            # Attempt initial connection in a background thread
            self.start_connection_manager()


    def start_connection_manager(self):
        """Starts a background thread to manage the connection (connect/reconnect)."""
        if self._connection_thread and self._connection_thread.is_alive():
            logger.info("Connection manager thread already running.")
            return

        logger.info("Starting UE connection manager thread.")
        self._stop_connection_event.clear()
        self._connection_thread = threading.Thread(target=self._connection_loop, name="UEConnectionManager", daemon=True)
        self._connection_thread.start()

    def _connection_loop(self):
        """Background loop to establish and maintain connection to UE."""
        attempts = 0
        while not self._stop_connection_event.is_set():
            if not self.is_connected:
                logger.info(f"Attempting to connect to UE ({attempts + 1}/{self.MAX_RECONNECT_ATTEMPTS if self.MAX_RECONNECT_ATTEMPTS > 0 else 'infinite'})...")
                if self._connect_socket():
                    logger.info("Successfully connected to UE. Starting receive thread.")
                    self._start_receive_thread() # Start receive thread upon successful connection
                    attempts = 0 # Reset attempts on successful connection
                else:
                    attempts += 1
                    if self.MAX_RECONNECT_ATTEMPTS > 0 and attempts >= self.MAX_RECONNECT_ATTEMPTS:
                        logger.error(f"Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. Stopping connection attempts.")
                        break # Exit loop

                    # Wait before next attempt
                    wait_time = self.RECONNECT_INTERVAL_SEC * (2 ** min(attempts, 4)) # Exponential backoff
                    logger.info(f"Connection failed. Retrying in {wait_time} seconds...")
                    self._stop_connection_event.wait(timeout=wait_time) # Wait or until stop is signaled
            else:
                # Connection is active, just sleep and check periodically
                # Or, implement a keep-alive mechanism if UE server supports it
                self._stop_connection_event.wait(timeout=self.RECONNECT_INTERVAL_SEC * 2) # Longer sleep when connected

        logger.info("UE connection manager thread stopped.")


    def _connect_socket(self):
        """Internal method to establish a socket connection."""
        with self._socket_lock: # Ensure only one thread tries to connect/disconnect socket
            if self.is_connected and self.socket:
                return True # Already connected

            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5) # Connection attempt timeout
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None) # Set to blocking for receive thread (or very long)
                                             # Individual send/receive ops can override this if needed.
                self.is_connected = True
                logger.info(f"Socket connected to UE at {self.host}:{self.port}")
                # Optionally: Trigger a "connected" event or callback
                return True
            except Exception as e:
                logger.warning(f"Socket connection to UE failed: {e}")
                self._cleanup_socket() # Ensure socket is cleaned up
                return False

    def _cleanup_socket(self):
        """Cleans up the socket object."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e_close:
                logger.error(f"Error closing UE socket: {e_close}")
            finally:
                self.socket = None
        self.is_connected = False
        # logger.debug("UE socket cleaned up.")


    def disconnect(self):
        """Public method to explicitly disconnect and stop threads."""
        logger.info("Disconnecting from Unreal Simulator and stopping threads...")
        self._stop_connection_event.set() # Signal connection manager thread to stop
        self._stop_receive_event.set()   # Signal receive thread to stop

        with self._socket_lock: # Protect socket during disconnect
            self._cleanup_socket()

        # Wait for threads to finish
        if self._connection_thread and self._connection_thread.is_alive():
            logger.debug("Waiting for UE connection manager thread to join...")
            self._connection_thread.join(timeout=self.RECONNECT_INTERVAL_SEC + 1)
            if self._connection_thread.is_alive(): logger.warning("UE connection manager thread did not join.")
        self._connection_thread = None

        if self._receive_thread and self._receive_thread.is_alive():
            logger.debug("Waiting for UE receive thread to join...")
            self._receive_thread.join(timeout=2.0) # Short timeout for receive thread
            if self._receive_thread.is_alive(): logger.warning("UE receive thread did not join.")
        self._receive_thread = None

        logger.info("Unreal Simulator Interface disconnected.")


    def _start_receive_thread(self):
        """Starts the background thread for receiving data from UE."""
        if self._receive_thread and self._receive_thread.is_alive():
            logger.debug("UE receive thread already running.")
            return
        self._stop_receive_event.clear()
        self._receive_thread = threading.Thread(target=self._receive_loop, name="UEReceiveLoop", daemon=True)
        self._receive_thread.start()
        logger.info("UE data receive thread started.")


    def _receive_loop(self):
        """Continuously receives data from UE and puts it on the response queue or calls callbacks."""
        logger.info("UE receive loop started.")
        buffer_size = 4096
        while not self._stop_receive_event.is_set() and self.is_connected:
            try:
                with self._socket_lock: # Ensure socket isn't closed by another thread while using
                    if not self.is_connected or not self.socket: break # Exit if disconnected

                    # 1. Read message size (4 bytes)
                    # Set a timeout for recv operations to allow checking stop_event
                    # This is crucial if socket is blocking.
                    self.socket.settimeout(0.1) # Short timeout for recv attempts
                    try:
                        raw_msglen = self.socket.recv(4)
                    except socket.timeout:
                        continue # Timeout, check stop_event and continue loop
                    except Exception as e_recv_size:
                         logger.error(f"UE receive loop: Error receiving size: {e_recv_size}. Assuming disconnect.")
                         self.is_connected = False # Mark as disconnected
                         break # Exit loop

                    if not raw_msglen:
                        logger.warning("UE receive loop: Connection closed by peer (received empty size).")
                        self.is_connected = False
                        break
                    msglen = int.from_bytes(raw_msglen, 'big')

                    # 2. Read message payload
                    self.socket.settimeout(config.UE_RESPONSE_TIMEOUT_SEC) # Longer timeout for payload
                    payload = b''
                    while len(payload) < msglen:
                        chunk = self.socket.recv(min(msglen - len(payload), buffer_size))
                        if not chunk:
                            logger.warning("UE receive loop: Connection closed by peer (received empty payload chunk).")
                            self.is_connected = False
                            break
                        payload += chunk
                    if not self.is_connected: break # Check again if disconnected during chunk receive


                # 3. Process message (JSON parsing)
                if payload:
                    try:
                        response_str = payload.decode('utf-8')
                        response_data = json.loads(response_str)
                        request_id = response_data.get('sequence_id') # Python request sends this
                        ue_action = response_data.get('ue_action') # UE might push unsolicited data with its own action type

                        if request_id is not None and request_id in self._response_wait_events:
                            # This is a response to a specific request
                            self._response_queue.put((request_id, response_data))
                            if self._response_wait_events[request_id].is_set(): # Check if event still exists
                                 self._response_wait_events[request_id].set() # Signal the waiting thread
                        elif ue_action is not None:
                            # This is an unsolicited push message from UE
                            logger.debug(f"Received unsolicited push from UE: Action='{ue_action}'")
                            for callback in self._push_data_callbacks:
                                try:
                                    callback(ue_action, response_data)
                                except Exception as cb_e:
                                    logger.error(f"Error in push data callback: {cb_e}")
                        else:
                            logger.warning(f"Received message from UE without known request_id or ue_action: {response_data}")

                    except json.JSONDecodeError:
                        logger.error(f"UE receive loop: Failed to decode JSON: {response_str[:200]}")
                    except Exception as e_proc:
                        logger.error(f"UE receive loop: Error processing message: {e_proc}")
                else:
                    logger.warning("UE receive loop: Received empty payload after size.")


            except Exception as e_loop:
                if self.is_connected: # Only log error if we thought we were connected
                    logger.error(f"UE receive loop: Unhandled exception: {e_loop}", exc_info=True)
                self.is_connected = False # Assume disconnect on major error
                break # Exit loop

        # Loop finished, clean up connection state if not already done
        with self._socket_lock:
            self._cleanup_socket()
        logger.info("UE receive loop stopped.")


    def _send_request(self, request_data, expect_response=True, timeout=None):
        """Internal helper to send a request and optionally wait for a response."""
        if not self.is_connected:
            logger.error(f"Cannot send request, not connected to UE. Action: {request_data.get('action')}")
            return None if expect_response else False

        request_id = str(uuid.uuid4()) # Generate a unique request ID
        request_data['sequence_id'] = request_id
        if expect_response:
            self._response_wait_events[request_id] = threading.Event()

        sent_successfully = False
        with self._socket_lock: # Ensure socket is valid for sending
            if not self.is_connected or not self.socket:
                logger.error(f"Cannot send request, connection lost before sending. Action: {request_data.get('action')}")
                if expect_response and request_id in self._response_wait_events:
                    del self._response_wait_events[request_id] # Clean up event
                return None if expect_response else False

            try:
                message = json.dumps(request_data).encode('utf-8')
                message_len = len(message)
                self.socket.sendall(message_len.to_bytes(4, 'big'))
                self.socket.sendall(message)
                sent_successfully = True
                logger.debug(f"Sent request to UE (ID: {request_id}): Action='{request_data.get('action')}'")
            except Exception as e:
                logger.error(f"Error sending request (ID: {request_id}) to UE: {e}. Assuming disconnect.")
                self.is_connected = False # Mark as disconnected
                if expect_response and request_id in self._response_wait_events:
                    del self._response_wait_events[request_id]
                return None if expect_response else False

        if not sent_successfully:
             return None if expect_response else False


        if expect_response:
            wait_event = self._response_wait_events[request_id]
            # Wait for the response to arrive (signaled by receive_loop)
            # Use config.UE_RESPONSE_TIMEOUT_SEC if no specific timeout provided
            effective_timeout = timeout if timeout is not None else config.UE_RESPONSE_TIMEOUT_SEC
            if wait_event.wait(timeout=effective_timeout):
                # Event was set, try to get response from queue
                # This assumes receive_loop puts (req_id, data) onto _response_queue
                # For true async, one might use futures or a more complex callback system.
                try:
                    # Search queue for our specific response (this is simplified)
                    # A better way is for receive_loop to directly fulfill a Future or Promise for this request_id.
                    # Or, have a dedicated response dict: self._responses_received[request_id] = data
                    q_timeout = 0.1 # Short timeout to get from queue
                    found_response = None
                    temp_holder = [] # To temporarily hold other responses
                    while True:
                        try:
                             resp_id, resp_data = self._response_queue.get(timeout=q_timeout)
                             if resp_id == request_id:
                                 found_response = resp_data
                                 break
                             else:
                                 temp_holder.append((resp_id, resp_data)) # Put back other responses
                        except queue.Empty:
                             logger.warning(f"Response event set for ID {request_id}, but not found in queue after timeout.")
                             break # Response not found in queue
                    # Put back any other responses read from the queue
                    for item in temp_holder: self._response_queue.put(item)

                    if found_response:
                         logger.debug(f"Received response for ID {request_id}: {found_response.get('status')}")
                         # Check status from UE response
                         if found_response.get('status') == 'success':
                             return found_response.get('results') or found_response.get('data') # UE might use 'results' or 'data'
                         else:
                             logger.error(f"UE returned error for request ID {request_id}: {found_response.get('error', 'Unknown UE error')}")
                             return None # Or return error dict: found_response
                    else:
                         return None # Event set but response not in queue, or timeout handling

                except Exception as e_q:
                     logger.error(f"Error getting response from queue for ID {request_id}: {e_q}")
                     return None
                finally:
                    if request_id in self._response_wait_events:
                         del self._response_wait_events[request_id] # Clean up event
            else:
                # Wait timed out
                logger.warning(f"Timeout waiting for response from UE for request ID {request_id} (Action: {request_data.get('action')}).")
                if request_id in self._response_wait_events:
                     del self._response_wait_events[request_id]
                return None # Or a specific timeout error object
        else:
            # Not expecting response (send-and-forget)
            return True # Indicates message was sent


    def register_push_data_callback(self, callback_func):
        """Registers a callback function to be called when unsolicited data is pushed from UE."""
        if callable(callback_func):
            self._push_data_callbacks.append(callback_func)
            logger.info(f"Registered push data callback: {callback_func.__name__}")
        else:
            logger.error("Provided push data callback is not callable.")


    # --- Public Interface Methods ---
    def run_simulation(self, welding_params, timeout=None):
        request_data = {'action': 'run_simulation', 'parameters': welding_params}
        return self._send_request(request_data, expect_response=True, timeout=timeout)

    def send_robot_pose(self, robot_id, joint_angles=None, tcp_transform=None):
        # Send-and-forget, no response expected.
        parameters = {"robot_id": robot_id, "joint_angles": joint_angles, "tcp_transform": tcp_transform}
        request_data = {"action": "set_robot_pose", "parameters": parameters}
        return self._send_request(request_data, expect_response=False) # Returns True if send was successful

    def send_welding_visual_command(self, robot_id, command_type, details=None):
        # Send-and-forget
        parameters = {"robot_id": robot_id, "command_type": command_type, "details": details}
        request_data = {"action": "welding_visual_command", "parameters": parameters}
        return self._send_request(request_data, expect_response=False)

    def get_sim2real_ark_situation(self, parameters=None, timeout=None):
        request_data = {'action': 'get_sim2real_ark_situation', 'parameters': parameters or {}}
        return self._send_request(request_data, expect_response=True, timeout=timeout)

    # --- Batch Simulation Request (Conceptual) ---
    def run_simulation_batch(self, parameter_sets: list[dict], timeout_per_sim=None, overall_timeout=None):
        """
        Sends a batch of simulation parameter sets to UE.
        UE should process these and return results, possibly in batches or individually.
        This is a complex interaction and requires careful protocol design with UE.
        """
        if not self.is_connected: return None
        # Protocol for batch simulation:
        # 1. Python sends a "start_batch_simulation" request with all parameter_sets.
        # 2. UE acknowledges and starts processing.
        # 3. UE sends individual simulation results (or batches of results) as they complete,
        #    each tagged with an ID corresponding to the input parameter set.
        # 4. Python's _receive_loop collects these results and uses callbacks or another queue.
        # This example simplifies by sending one request and expecting one aggregated response,
        # which might not be feasible for very large batches or long simulations.

        request_data = {'action': 'run_simulation_batch', 'parameters': {'sim_sets': parameter_sets}}
        logger.info(f"Requesting batch simulation for {len(parameter_sets)} sets.")
        # The timeout here should be long enough for all simulations.
        # A better approach would be UE sending progress updates or individual results.
        return self._send_request(request_data, expect_response=True, timeout=overall_timeout)


# Example Usage (run in a separate script or if __name__ == '__main__')
# Ensure a dummy UE server is running that can handle these actions and sequence_ids.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    logger.info("--- Advanced Physics Interface Example ---")

    # Dummy config for testing
    class DummyConfig:
        UE_SIMULATOR_IP = '127.0.0.1'
        UE_SIMULATOR_PORT = 9999
        UE_RESPONSE_TIMEOUT_SEC = 10
        UE_RECONNECT_INTERVAL_SEC = 3
        UE_MAX_RECONNECT_ATTEMPTS = 2
    config = DummyConfig()

    # --- Dummy UE Server (More advanced to handle sequence_id and push) ---
    def advanced_dummy_ue_server(host, port, stop_event):
        # (Server implementation needs to be more robust for this client)
        # ... (server code from previous physics_interface example, modified to echo sequence_id) ...
        logger.info(f"[AdvancedDummyUEServer] Starting on {host}:{port}")
        # This dummy server needs to be implemented to work with the client's async logic
        # For now, just a placeholder that won't fully test the client.
        # A real test would require a server that:
        # 1. Echoes sequence_id in responses.
        # 2. Potentially pushes unsolicited messages with "ue_action".
        time.sleep(1) # Simulate server startup
        logger.info("[AdvancedDummyUEServer] Exiting (not fully implemented for this client example).")


    # --- Example Push Data Callback ---
    def my_ue_push_handler(action, data):
        logger.info(f"HMI/System received PUSH from UE: Action='{action}', Data='{data}'")

    # --- Initialize and Test ---
    ue_interface = UnrealSimulatorInterface() # Starts connection manager

    # Register a callback for pushed data
    ue_interface.register_push_data_callback(my_ue_push_handler)

    logger.info("Waiting for UE connection (check connection_loop)...")
    # Loop to check connection status (connection_loop runs in background)
    for _ in range(config.UE_MAX_RECONNECT_ATTEMPTS + 2): # Wait a bit longer
        if ue_interface.is_connected:
            logger.info("UE Interface is connected!")
            break
        time.sleep(config.UE_RECONNECT_INTERVAL_SEC / 2)
    else:
        logger.error("UE Interface failed to connect after several attempts.")
        ue_interface.disconnect() # Ensure threads are stopped
        sys.exit(1)


    # Test send_robot_pose (send-and-forget)
    logger.info("\n--- Testing send_robot_pose ---")
    if ue_interface.send_robot_pose(robot_id=1, joint_angles=[10,20,30,40,50,60]):
        logger.info("send_robot_pose request sent.")
    else:
        logger.error("Failed to send robot_pose request.")


    # Test run_simulation (waits for response)
    logger.info("\n--- Testing run_simulation ---")
    sim_params = {'current': 150, 'voltage': 22}
    results = ue_interface.run_simulation(sim_params, timeout=5) # Shorter timeout for example
    if results:
        logger.info(f"Simulation results: {results}")
    else:
        logger.warning("Did not receive simulation results or call failed.")

    # Test batch simulation (conceptual)
    logger.info("\n--- Testing run_simulation_batch ---")
    batch_params = [
        {'current': 160, 'voltage': 23, 'id': 'sim_A'},
        {'current': 170, 'voltage': 24, 'id': 'sim_B'}
    ]
    batch_results = ue_interface.run_simulation_batch(batch_params, overall_timeout=10)
    if batch_results:
        logger.info(f"Batch simulation results: {batch_results}")
    else:
        logger.warning("Did not receive batch simulation results or call failed.")


    # Wait a bit for potential push messages if dummy server supported it
    logger.info("Waiting a few seconds for any potential push messages from UE...")
    time.sleep(3)

    # Shutdown
    logger.info("\n--- Shutting down UE Interface ---")
    ue_interface.disconnect() # Stops connection and receive threads

    logger.info("--- Advanced Physics Interface Example Finished ---")