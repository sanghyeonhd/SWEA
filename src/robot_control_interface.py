# robot_control_interface.py (Continued)

import threading
import time
import json
import socket
import logging
from src import config

logger = logging.getLogger(__name__)

class RobotControlInterface:
    def __init__(self, config=None):
        self.config = config
        self.connections = {}
        self.connected_robot_ids = []
        # 실제 연결 초기화 등 필요한 코드 추가

    def _get_connection(self, robot_id):
        # 실제 연결 객체 반환 (더미)
        return self.connections.get(robot_id)

    def set_welding_parameters(self, robot_id, current, voltage, speed):
        """
        Sends commands to set welding parameters (current, voltage, speed).
        This might be done via setting specific robot registers or calling a dedicated function.
        """
        conn = self._get_connection(robot_id)
        if conn:
            # This is highly dependent on how the actual robot/welder interface is designed.
            # Option 1: Set individual parameters via "set_param" action
            params_to_set = {
                'welding_current': current,
                'welding_voltage': voltage,
                'welding_speed': speed # Robot's motion speed for the welding path
            }
            all_success = True
            for param_name, value in params_to_set.items():
                response = conn.send_command_and_wait_response("set_param", parameters={'param_name': param_name, 'value': value})
                if not (response and response.get('status') == 'success'):
                    logger.error(f"Robot {robot_id}: Failed to set parameter '{param_name}' to {value}. Response: {response}")
                    all_success = False
                    # Break or continue depends on whether partial success is acceptable
                    break

            if all_success:
                logger.info(f"Robot {robot_id}: Welding parameters (current, voltage, speed) set successfully.")
                return True
            else:
                logger.error(f"Robot {robot_id}: Failed to set some or all welding parameters.")
                return False
        return False

    def set_robot_speed_override(self, robot_id, speed_percentage):
        """Sets the robot's global speed override (e.g., 0-100%)."""
        conn = self._get_connection(robot_id)
        if conn:
             if not (0 <= speed_percentage <= 100):
                 logger.error(f"Robot {robot_id}: Invalid speed override percentage: {speed_percentage}. Must be between 0 and 100.")
                 return False
             response = conn.send_command_and_wait_response("set_param", parameters={'param_name': 'speed_override', 'value': speed_percentage})
             if response and response.get('status') == 'success':
                 logger.info(f"Robot {robot_id}: Speed override set to {speed_percentage}%.")
                 return True
             elif response:
                 logger.error(f"Robot {robot_id}: Failed to set speed override. Response: {response}")
             return False


    # --- Example: Asynchronous Status Monitoring (Conceptual) ---
    # In a real system, status updates might be pushed by the robot or polled periodically.
    # This is a very simplified polling example.
    def start_status_monitoring(self, robot_id, interval_seconds=5):
        """Starts a background thread to periodically poll robot status."""
        conn = self._get_connection(robot_id)
        if not conn:
            return

        if hasattr(conn, '_monitoring_thread') and conn._monitoring_thread.is_alive():
            logger.warning(f"Robot {robot_id}: Status monitoring is already active.")
            return

        def _monitor_loop():
            logger.info(f"Robot {conn.robot_id}: Starting status monitor (interval: {interval_seconds}s).")
            while conn.is_connected: # Loop while connected
                status_data = self.get_robot_status(conn.robot_id) # Uses the public method
                if status_data:
                    # In a real application, this data would be published to a queue,
                    # used to update a UI, or logged.
                    logger.info(f"Robot {conn.robot_id} Status Update: {status_data}")
                else:
                    # If get_robot_status fails repeatedly, the connection might be dead.
                    # The RobotConnection object handles disconnect on errors.
                    pass
                time.sleep(interval_seconds)
            logger.info(f"Robot {conn.robot_id}: Status monitor stopped.")

        conn._monitoring_active = True # A flag to control the loop from outside if needed
        conn._monitoring_thread = threading.Thread(target=_monitor_loop, daemon=True)
        conn._monitoring_thread.start()

    def stop_status_monitoring(self, robot_id):
        """Stops the background status monitoring for a robot."""
        conn = self.connections.get(robot_id)
        if conn and hasattr(conn, '_monitoring_thread') and conn._monitoring_thread.is_alive():
            logger.info(f"Robot {robot_id}: Stopping status monitor...")
            conn.is_connected = False # Signal the loop in _monitor_loop to exit
            conn._monitoring_thread.join(timeout=5) # Wait for thread to finish
            if conn._monitoring_thread.is_alive():
                logger.warning(f"Robot {robot_id}: Monitoring thread did not terminate gracefully.")
            else:
                logger.info(f"Robot {robot_id}: Status monitor stopped.")
        else:
            logger.info(f"Robot {robot_id}: Status monitoring not active or connection not found.")


# Example Usage (requires a dummy robot server for each configured robot)
if __name__ == '__main__':
    logger.info("--- Robot Control Interface Example ---")

    # --- Create Dummy Config (for example purposes only) ---
    # In a real run, this would be imported from config.py
    class DummyConfig:
        ROBOT_CONFIGS = [
            {'id': 1, 'name': 'Robot1', 'ip': '127.0.0.1', 'port': 6001},
            # {'id': 2, 'name': 'Robot2', 'ip': '127.0.0.1', 'port': 6002},
        ]
    config = DummyConfig() # Use dummy config

    # --- Dummy Robot Server (Placeholder for actual robot controller) ---
    # This server simulates one robot listening for commands.
    # Run this in separate terminals for each robot for a more complete test.
    def dummy_robot_server(robot_id, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1)
            logger.info(f"[DummyRobot {robot_id}] Listening on {host}:{port}")

            while True:
                client_socket, addr = server_socket.accept()
                logger.info(f"[DummyRobot {robot_id}] Accepted connection from {addr}")
                try:
                    while True: # Handle multiple commands on one connection
                        # Receive message size
                        raw_msglen = client_socket.recv(4)
                        if not raw_msglen: break # Client disconnected
                        msglen = int.from_bytes(raw_msglen, 'big')

                        # Receive message
                        data = b''
                        while len(data) < msglen:
                            chunk = client_socket.recv(min(msglen - len(data), 4096))
                            if not chunk: break # Client disconnected
                            data += chunk
                        if not data: break

                        request = json.loads(data.decode('utf-8'))
                        logger.info(f"[DummyRobot {robot_id}] Received request: {request}")

                        # Process request and prepare response (simplified)
                        response_data = {"data": {}}
                        if request['action'] == "get_status":
                            response_data["status"] = "success"
                            response_data["data"] = {"position": [1.0, 2.0, 3.0], "job_running": "IDLE", "speed_override": 100}
                        elif request['action'] == "run_job":
                            response_data["status"] = "success"
                            response_data["message"] = f"Job {request['parameters']['job_name']} started."
                        elif request['action'] == "set_param":
                            response_data["status"] = "success"
                            response_data["message"] = f"Param {request['parameters']['param_name']} set to {request['parameters']['value']}."
                        else:
                            response_data["status"] = "failure"
                            response_data["error"] = "Unknown action"

                        response_data["sequence_id"] = request["sequence_id"]
                        response_message = json.dumps(response_data).encode('utf-8')
                        response_len = len(response_message)

                        client_socket.sendall(response_len.to_bytes(4, 'big'))
                        client_socket.sendall(response_message)
                        logger.info(f"[DummyRobot {robot_id}] Sent response for Seq ID {request['sequence_id']}")

                except ConnectionResetError:
                     logger.warning(f"[DummyRobot {robot_id}] Connection reset by client.")
                except Exception as e_inner:
                    logger.error(f"[DummyRobot {robot_id}] Error handling client: {e_inner}")
                finally:
                    logger.info(f"[DummyRobot {robot_id}] Closing client socket from {addr}")
                    client_socket.close()
        except Exception as e_outer:
            logger.error(f"[DummyRobot {robot_id}] Server error: {e_outer}")
        finally:
            logger.info(f"[DummyRobot {robot_id}] Shutting down server.")
            server_socket.close()

    # Start dummy server(s) in separate threads for testing
    server_threads = []
    for cfg in config.ROBOT_CONFIGS:
        st = threading.Thread(target=dummy_robot_server, args=(cfg['id'], cfg['ip'], cfg['port']), daemon=True)
        server_threads.append(st)
        st.start()
        time.sleep(0.1) # Give server a moment to start

    if not server_threads:
        logger.error("No dummy servers started. Example will likely fail to connect.")


    # --- Initialize and Use the Interface ---
    robot_interface = RobotControlInterface()

    if robot_interface.connect_all():
        logger.info(f"Successfully connected to robots: {robot_interface.connected_robot_ids}")

        if robot_interface.connected_robot_ids:
            test_robot_id = robot_interface.connected_robot_ids[0] # Use the first connected robot

            # Test Get Status
            logger.info(f"\n--- Testing Get Status for Robot {test_robot_id} ---")
            status = robot_interface.get_robot_status(test_robot_id)
            if status:
                logger.info(f"Robot {test_robot_id} Status: {status}")

            # Test Run Job
            logger.info(f"\n--- Testing Run Job for Robot {test_robot_id} ---")
            if robot_interface.run_job(test_robot_id, "WELD_PROGRAM_001"):
                logger.info("Run job command sent successfully.")

            # Test Set Welding Parameters
            logger.info(f"\n--- Testing Set Welding Parameters for Robot {test_robot_id} ---")
            if robot_interface.set_welding_parameters(test_robot_id, current=155, voltage=22.5, speed=280):
                logger.info("Set welding parameters command sent successfully.")

            # Test Set Speed Override
            logger.info(f"\n--- Testing Set Speed Override for Robot {test_robot_id} ---")
            if robot_interface.set_robot_speed_override(test_robot_id, 80):
                logger.info("Set speed override command sent successfully.")

            # Test Status Monitoring (Optional, can be verbose)
            # logger.info(f"\n--- Testing Status Monitoring for Robot {test_robot_id} ---")
            # robot_interface.start_status_monitoring(test_robot_id, interval_seconds=2)
            # time.sleep(5) # Let it monitor for a bit
            # robot_interface.stop_status_monitoring(test_robot_id)

        robot_interface.disconnect_all()
    else:
        logger.warning("Failed to connect to any robots in the example.")

    # Wait for server threads to finish (they are daemons, so will exit with main)
    # but for clean logs, we might give them a moment if they were doing something
    # In a real app, server management would be more robust.
    logger.info("--- Example Finished ---")