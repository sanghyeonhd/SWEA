# src/physics_interface.py
# Description: Defines an interface to communicate with the Unreal Engine physics simulator/visualizer.
# Handles sending robot pose and welding visual commands to UE.

import json
import socket
import time
import logging # Use standard logging
import threading # For thread-safe socket operations if needed

import config

# Setup logging for this module
logger = logging.getLogger(__name__)


class UnrealSimulatorInterface:
    """
    Represents the connection to the Unreal Engine physics simulator/visualizer.
    Assumes UE is running a server listening for requests on TCP/IP.
    Communicates using 4-byte length prefix + JSON payload.
    """
    def __init__(self, host=config.UE_SIMULATOR_IP, port=config.UE_SIMULATOR_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self._lock = threading.Lock() # Protect socket operations if multiple threads access this instance
        self._sequence_id_counter = 0 # Basic sequence ID counter

        logger.info(f"Unreal Simulator Interface initialized for {self.host}:{self.port}")

    def connect(self):
        """Establishes connection to the UE simulator."""
        if self.is_connected and self.socket:
            logger.info(f"Already connected to Unreal Simulator at {self.host}:{self.port}")
            return True

        logger.info(f"Attempting to connect to Unreal Simulator at {self.host}:{self.port}")
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5) # Timeout for connection attempt
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(10) # Timeout for subsequent send/receive operations
            self.is_connected = True
            logger.info(f"Successfully connected to Unreal Simulator at {self.host}:{self.port}")
            # Optionally send an initial handshake/config message here
            # self.send_request({'action': 'handshake', 'version': '0.1'})
            return True
        except ConnectionRefusedError:
            logger.error(f"Connection refused. Is the Unreal simulation/visualization server running at {self.host}:{self.port}?")
            self._cleanup_connection()
            return False
        except socket.timeout:
             logger.error(f"Connection timed out attempting to reach {self.host}:{self.port}.")
             self._cleanup_connection()
             return False
        except Exception as e:
            logger.error(f"Error connecting to Unreal Simulator at {self.host}:{self.port}: {e}")
            self._cleanup_connection()
            return False

    def disconnect(self):
        """Closes the connection."""
        if self.is_connected and self.socket:
            logger.info("Disconnecting from Unreal Simulator.")
            try:
                # Attempt a graceful shutdown first
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except Exception as e:
                logger.error(f"Error during socket shutdown/close: {e}")
            finally:
                 self._cleanup_connection()
        else:
             logger.info("Not connected to Unreal Simulator.")

    def _cleanup_connection(self):
         """Internal helper to clean up socket state."""
         self.socket = None
         self.is_connected = False
         logger.info("Unreal Simulator connection state cleaned up.")


    def _send_framed_json(self, data):
        """Sends JSON data with 4-byte length prefix."""
        if not self.is_connected or not self.socket:
            # logger.warning("Not connected to simulator, cannot send data.") # Too frequent if polling
            return False
        try:
            message = json.dumps(data).encode('utf-8')
            message_len = len(message)
            # Ensure thread-safe sending if multiple threads might use the same socket
            with self._lock:
                 self.socket.sendall(message_len.to_bytes(4, 'big')) # Send message size (4 bytes, big endian)
                 self.socket.sendall(message) # Send the actual JSON message
            # logger.debug(f"Sent {message_len} bytes to UE.") # Enable for verbose debugging
            return True
        except socket.timeout:
            logger.error("Socket timeout during send. Disconnecting.")
            self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Error sending data to UE: {e}. Disconnecting.")
            self.disconnect()
            return False

    def _receive_framed_json(self, buffer_size=4096):
        """Receives JSON data with 4-byte length prefix."""
        if not self.is_connected or not self.socket:
            # logger.warning("Not connected to simulator, cannot receive data.") # Too frequent if polling
            return None # Return None immediately if not connected

        try:
            with self._lock: # Ensure thread-safe receiving
                # Read message size (blocking call, respects socket timeout)
                raw_msglen = self.socket.recv(4)
                if not raw_msglen: # Connection closed by peer
                    logger.warning("Connection closed by peer during size receive.")
                    self.disconnect()
                    return None
                msglen = int.from_bytes(raw_msglen, 'big')
                # logger.debug(f"Expecting {msglen} bytes from UE.") # Enable for verbose debugging

                # Read the actual message
                data = b''
                while len(data) < msglen:
                    # Read chunks until the full message is received
                    chunk = self.socket.recv(min(msglen - len(data), buffer_size))
                    if not chunk: # Connection closed by peer
                        logger.warning("Connection closed by peer during data receive.")
                        self.disconnect()
                        return None
                    data += chunk

            # Decode and parse the JSON message
            response = json.loads(data.decode('utf-8'))
            # logger.debug("Received data from UE.") # Enable for verbose debugging
            return response

        except socket.timeout:
            # Timeout is expected if waiting for a specific response that doesn't come
            # It doesn't necessarily mean disconnection, but might indicate UE is slow or stuck.
            # Depending on expected interaction, may or may not disconnect here.
            # For 'send and forget' visual updates, a receive timeout likely means no response
            # was expected or it's delayed, not necessarily a fatal error.
            # For 'send and wait for response' (like run_simulation), timeout IS a failure.
            logger.warning("Socket timeout during receive.")
            return {"status": "timeout", "message": "Receive timeout"} # Indicate timeout, but keep connection open

        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON response from UE: {data.decode('utf-8', errors='ignore')}")
            # Protocol error, might be recoverable, but for robustness, consider disconnecting
            self.disconnect()
            return None
        except Exception as e:
            logger.error(f"Error receiving data from UE: {e}. Disconnecting.")
            self.disconnect()
            return None

    def _generate_sequence_id(self):
        """Generates a simple sequence ID for requests that might expect a response."""
        # Using lock to ensure thread-safe counter increment if needed
        with self._lock:
             self._sequence_id_counter += 1
             return self._sequence_id_counter

    # --- Existing run_simulation method (potentially modified to expect response) ---
    def run_simulation(self, welding_params, timeout=30):
        """
        Sends welding parameters to UE, runs simulation, and gets results.
        Expects a response from UE.
        """
        if not self.is_connected:
            # Attempt to connect if not already
            if not self.connect():
                return None # Failed to connect

        seq_id = self._generate_sequence_id() # Generate sequence ID for this request
        request_data = {
            'action': 'run_simulation', # Action name must match UE server
            'parameters': welding_params,
            'sequence_id': seq_id # Include sequence ID
        }

        logger.info(f"Sending simulation request (Seq ID: {seq_id}) to UE.")
        if not self._send_framed_json(request_data):
            logger.error(f"Failed to send simulation request (Seq ID: {seq_id}).")
            return None # Send failed

        # Now wait for the response with the matching sequence ID
        # Note: A robust implementation would handle receiving messages out of order
        # or messages not intended as responses to this specific request.
        # This example assumes the NEXT message received is the response.
        # In a real-time system, this synchronous wait might not be suitable.
        logger.info(f"Waiting for simulation response (Seq ID: {seq_id}) from UE...")
        # Temporarily set a receive timeout for this specific response
        original_timeout = self.socket.gettimeout()
        try:
            self.socket.settimeout(timeout) # Set timeout for receiving response
            response = self._receive_framed_json()

            if response is None:
                 logger.error(f"Failed to receive response for simulation request (Seq ID: {seq_id}). Connection likely closed.")
                 return None
            elif response.get('sequence_id') != seq_id:
                 logger.warning(f"Received response with mismatching Seq ID. Expected {seq_id}, got {response.get('sequence_id')}. Response: {response}")
                 # Depending on protocol, might need to handle this unexpected message
                 # For simplicity, treat as failure to get the *correct* response
                 return None
            elif response.get('status') == 'success':
                logger.info(f"Simulation request (Seq ID: {seq_id}) successful on UE side.")
                return response.get('results') # Assuming results are in 'results' field
            elif response.get('status') == 'timeout':
                 logger.warning(f"Simulation request (Seq ID: {seq_id}) receive timed out.")
                 return {"status": "timeout", "message": "Receive timeout waiting for simulation result"}
            else: # Status is 'failure' or other error
                logger.error(f"Simulation request (Seq ID: {seq_id}) failed on UE side: {response.get('error', 'No details')}")
                return {"status": "failure", "error": response.get('error', 'Unknown error')}

        except socket.timeout:
             logger.error(f"Timeout waiting for simulation response (Seq ID: {seq_id}).")
             return {"status": "timeout", "message": "Response timeout"}
        except Exception as e:
             logger.error(f"Error while waiting for or processing simulation response (Seq ID: {seq_id}): {e}", exc_info=True)
             return {"status": "failure", "error": f"Internal error processing response: {e}"}
        finally:
             # Restore original socket timeout
             if self.is_connected and self.socket:
                  self.socket.settimeout(original_timeout)


    # --- NEW Method: send_robot_pose ---
    def send_robot_pose(self, robot_id, joint_angles=None, tcp_transform=None):
        """
        Sends robot pose data (joint angles or TCP transform) to UE for visualization.
        Does NOT necessarily wait for a response (send-and-forget for real-time streaming).
        """
        if not self.is_connected:
            # logger.warning("Not connected to UE simulator, cannot send robot pose.") # Too frequent
            return False # Indicate message was not sent

        # Choose one pose representation to send based on what's provided
        if joint_angles is None and tcp_transform is None:
            logger.warning(f"No pose data (joint_angles or tcp_transform) provided for Robot {robot_id}.")
            return False

        parameters = {
            "robot_id": robot_id, # Include robot_id in parameters for UE to find the actor
            "joint_angles": joint_angles, # Can be None
            "tcp_transform": tcp_transform # Can be None
        }

        request_data = {
            "action": "set_robot_pose", # Action name must match UE server
            "parameters": parameters,
            # For real-time streaming, sequence ID might not be needed if no response is expected
            # sequence_id: self._generate_sequence_id() # Optional
        }

        # logger.debug(f"Sending pose for Robot {robot_id} to UE: {joint_angles or tcp_transform}") # Enable for verbose logging
        # Use _send_framed_json and check if it succeeded
        success = self._send_framed_json(request_data)
        if not success:
             # Error already logged inside _send_framed_json
             pass # No further action needed here for failure to send
        return success # True if sent, False otherwise


    # --- NEW Method: send_welding_visual_command ---
    def send_welding_visual_command(self, robot_id, command_type, details=None):
        """
        Sends commands related to welding visualization (e.g., Arc On/Off) to UE.
        Does NOT necessarily wait for a response (send-and-forget).
        """
        if not self.is_connected:
            # logger.warning("Not connected to UE simulator, cannot send welding visual command.")
            return False # Indicate message was not sent

        parameters = {
            "robot_id": robot_id, # Include robot_id
            "command_type": command_type, # e.g., "arc_on", "arc_off", "set_arc_color"
            "details": details # e.g., {'color': [1.0, 0.5, 0.0]}
        }

        request_data = {
            "action": "welding_visual_command", # Action name must match UE server
            "parameters": parameters,
            # sequence_id: self._generate_sequence_id() # Optional
        }

        logger.debug(f"Sending welding visual command '{command_type}' for Robot {robot_id} to UE.")
        # Use _send_framed_json and check if it succeeded
        success = self._send_framed_json(request_data)
        if not success:
             # Error already logged inside _send_framed_json
             pass # No further action needed here for failure to send
        return success # True if sent, False otherwise


    # --- Optional: Add method to request visual feedback (e.g., camera image) ---
    # def request_camera_image(self, robot_id, camera_name, timeout=5):
    #     """Sends request to UE to get an image from a simulated camera."""
    #     if not self.is_connected: return None
    #     seq_id = self._generate_sequence_id()
    #     request_data = {
    #         "action": "get_camera_image",
    #         "robot_id": robot_id,
    #         "parameters": {"camera_name": camera_name},
    #         "sequence_id": seq_id
    #     }
    #     logger.info(f"Requesting camera image from Robot {robot_id} (Seq ID: {seq_id}).")
    #     if not self._send_framed_json(request_data): return None
    #     # This requires a more complex receive logic or a separate receiving thread
    #     # if the response might be large or arrive asynchronously.
    #     # For synchronous request/response:
    #     # response = self._receive_framed_json(timeout=timeout)
    #     # Check sequence ID, status, and parse image data (base64? binary? file path?)
    #     # return parsed_image_data or None
    #     logger.warning("request_camera_image is not fully implemented.")
    #     return None # Placeholder


    # --- Placeholder for Sim2Real Ark Situation (Keep or refine) ---
    # This method's purpose needs clarification - is it asking UE for a simulated
    # "Ark Situation" to compare with real? Or sending real to UE?
    # Based on the name, it sounds like requesting a representation from UE.
    def get_sim2real_ark_situation(self, parameters=None, timeout=5):
        """Placeholder to get Sim2Real Ark situation representation from UE."""
        if not self.is_connected: return None
        seq_id = self._generate_sequence_id()
        request_data = {
            "action": "get_sim2real_ark_situation",
            "parameters": parameters if parameters is not None else {},
            "sequence_id": seq_id
        }
        logger.info(f"Requesting Sim2Real Ark Situation from UE (Seq ID: {seq_id}).")
        if not self._send_framed_json(request_data): return None

        logger.info(f"Waiting for Sim2Real Ark Situation response (Seq ID: {seq_id})...")
        original_timeout = self.socket.gettimeout()
        try:
            self.socket.settimeout(timeout)
            response = self._receive_framed_json()
            if response and response.get('sequence_id') == seq_id and response.get('status') == 'success':
                 logger.info(f"Sim2Real Ark Situation request (Seq ID: {seq_id}) successful.")
                 return response.get('data') # Assuming data is in 'data' field
            elif response and response.get('sequence_id') == seq_id:
                 logger.error(f"Sim2Real Ark Situation request (Seq ID: {seq_id}) failed on UE side: {response.get('error', 'No details')}")
                 return None
            elif response:
                 logger.warning(f"Received mismatching Seq ID for Sim2Real request. Expected {seq_id}, got {response.get('sequence_id')}. Response: {response}")
                 return None
            else:
                 logger.error(f"Failed to receive valid response for Sim2Real request (Seq ID: {seq_id}).")
                 return None

        except socket.timeout:
            logger.warning(f"Timeout waiting for Sim2Real Ark Situation response (Seq ID: {seq_id}).")
            return {"status": "timeout", "message": "Response timeout"}
        except Exception as e:
            logger.error(f"Error processing Sim2Real Ark Situation response (Seq ID: {seq_id}): {e}", exc_info=True)
            return None
        finally:
             if self.is_connected and self.socket:
                 self.socket.settimeout(original_timeout)


# Example Usage (requires a matching UE server running)
# This __main__ block is for testing this specific module in isolation.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Use the module-level logger created at the top
    logger.info("--- Unreal Simulator Interface Example ---")

    # --- Dummy Config (for example purposes only) ---
    class DummyConfig:
        UE_SIMULATOR_IP = '127.0.0.1'
        UE_SIMULATOR_PORT = 9999
    config = DummyConfig()

    # --- Dummy UE Server (Placeholder for actual Unreal Engine application) ---
    # You would run this in a separate process or terminal for testing.
    # This simulates a very basic UE server receiving and acknowledging messages.
    def dummy_ue_server(host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1)
            logger.info(f"[DummyUEServer] Listening on {host}:{port}")

            conn, addr = server_socket.accept()
            logger.info(f"[DummyUEServer] Accepted connection from {addr}")
            conn.settimeout(5) # Set a timeout for receiving from client

            with conn:
                while True:
                    try:
                         # Receive message size
                        raw_msglen = conn.recv(4)
                        if not raw_msglen: break # Client disconnected
                        msglen = int.from_bytes(raw_msglen, 'big')

                        # Receive message
                        data = b''
                        while len(data) < msglen:
                            chunk = conn.recv(min(msglen - len(data), 4096))
                            if not chunk: break # Client disconnected
                            data += chunk
                        if not data: break

                        request = json.loads(data.decode('utf-8'))
                        logger.info(f"[DummyUEServer] Received request: {request}")

                        # --- Simulate processing and sending response ---
                        response_data = {"status": "success", "sequence_id": request.get("sequence_id"), "message": "Acknowledged"}
                        if request.get("action") == "run_simulation":
                            # Simulate simulation taking time and providing results
                            logger.info("[DummyUEServer] Simulating run_simulation...")
                            time.sleep(1) # Simulate work
                            response_data["results"] = {"predicted_bead_shape": "DummyGoodBead", "quality_score": 0.9}
                            logger.info("[DummyUEServer] Simulation done.")
                        elif request.get("action") == "set_robot_pose":
                             # Just acknowledge pose updates
                             pass
                        elif request.get("action") == "welding_visual_command":
                             # Just acknowledge visual commands
                             pass
                        elif request.get("action") == "get_sim2real_ark_situation":
                             logger.info("[DummyUEServer] Simulating get_sim2real_ark_situation...")
                             time.sleep(0.5)
                             response_data["data"] = {"simulated_state": {"temp": 380, "arc_power": 5000}}
                             logger.info("[DummyUEServer] Sim2Real data ready.")


                        # Send response
                        response_message = json.dumps(response_data).encode('utf-8')
                        response_len = len(response_message)
                        conn.sendall(response_len.to_bytes(4, 'big'))
                        conn.sendall(response_message)
                        logger.info(f"[DummyUEServer] Sent response for Seq ID {request.get('sequence_id')}")

                    except socket.timeout:
                        logger.debug("[DummyUEServer] Receive timeout, waiting for next message.")
                        continue # Keep waiting
                    except ConnectionResetError:
                        logger.warning("[DummyUEServer] Client disconnected unexpectedly.")
                        break # Exit inner loop
                    except Exception as e_inner:
                        logger.error(f"[DummyUEServer] Error handling client request: {e_inner}", exc_info=True)
                        # Send an error response if possible before breaking
                        try:
                             err_response = {"status": "failure", "error": f"Server internal error: {e_inner}", "sequence_id": request.get("sequence_id")}
                             err_message = json.dumps(err_response).encode('utf-8')
                             conn.sendall(len(err_message).to_bytes(4, 'big'))
                             conn.sendall(err_message)
                        except: pass # Ignore errors sending error response
                        break # Exit inner loop

        except Exception as e_outer:
            logger.error(f"[DummyUEServer] Server error: {e_outer}", exc_info=True)
        finally:
            logger.info("[DummyUEServer] Shutting down server.")
            server_socket.close()


    # --- Start Dummy Server Thread ---
    # Run this in a separate thread so the main script can use the client
    server_thread = threading.Thread(target=dummy_ue_server, args=(config.UE_SIMULATOR_IP, config.UE_SIMULATOR_PORT), daemon=True)
    server_thread.start()
    time.sleep(0.5) # Give server a moment to start


    # --- Initialize and Test the Interface ---
    simulator_interface = UnrealSimulatorInterface()

    # Test Connection
    if simulator_interface.connect():
        # Test send_robot_pose (send-and-forget)
        logger.info("\n--- Testing send_robot_pose ---")
        dummy_pose = {'joint_angles': [10.5, -5.2, 90.0, 0.0, 5.0, -10.0]}
        simulator_interface.send_robot_pose(robot_id=1, joint_angles=dummy_pose['joint_angles'])
        # No explicit wait for response here

        # Test send_welding_visual_command (send-and-forget)
        logger.info("\n--- Testing send_welding_visual_command ---")
        simulator_interface.send_welding_visual_command(robot_id=1, command_type="arc_on", details={'current': 160})
        time.sleep(0.1) # Give a moment for message to send/process
        simulator_interface.send_welding_visual_command(robot_id=1, command_type="arc_off")
        time.sleep(0.1)


        # Test run_simulation (send-and-wait-for-response)
        logger.info("\n--- Testing run_simulation ---")
        test_params = {'current': 150, 'voltage': 22, 'speed': 300}
        sim_results = simulator_interface.run_simulation(test_params)
        if sim_results:
            logger.info(f"run_simulation results: {sim_results}")
        else:
            logger.error("run_simulation failed.")

        # Test get_sim2real_ark_situation (send-and-wait-for-response)
        logger.info("\n--- Testing get_sim2real_ark_situation ---")
        sim2real_data = simulator_interface.get_sim2real_ark_situation({'request_type': 'latest'})
        if sim2real_data and sim2real_data.get('status') != 'timeout': # Check if it's not just a timeout response
             logger.info(f"get_sim2real_ark_situation data: {sim2real_data}")
        elif sim2real_data and sim2real_data.get('status') == 'timeout':
             logger.warning("get_sim2real_ark_situation timed out.")
        else:
             logger.error("get_sim2real_ark_situation failed.")


        # Test Disconnection
        simulator_interface.disconnect()
    else:
        logger.error("Failed to connect to the dummy UE simulator.")

    # Allow time for server thread to potentially log final messages before main thread exits
    time.sleep(1.0)
    logger.info("--- Unreal Simulator Interface Example Finished ---")