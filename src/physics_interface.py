# physics_interface.py
# Description: Defines an interface to communicate with the Unreal Engine physics simulator.
# This is a placeholder and requires actual communication implementation (e.g., socket).

import json
import socket
import time
import config

class UnrealSimulatorInterface:
    """
    Represents the connection to the Unreal Engine physics simulator.
    Assumes UE is running a server listening for requests.
    """
    def __init__(self, host=config.UE_SIMULATOR_IP, port=config.UE_SIMULATOR_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False

    def connect(self):
        """Establishes connection to the UE simulator."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(10) # Set a timeout for operations
            self.is_connected = True
            print(f"Successfully connected to Unreal Simulator at {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"Error: Connection refused. Is the Unreal simulation server running at {self.host}:{self.port}?")
            self.socket = None
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Error connecting to Unreal Simulator: {e}")
            self.socket = None
            self.is_connected = False
            return False

    def disconnect(self):
        """Closes the connection."""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.is_connected = False
            print("Disconnected from Unreal Simulator.")

    def send_request(self, data):
        """Sends data (e.g., welding parameters) to UE."""
        if not self.is_connected:
            print("Error: Not connected to the simulator.")
            return None
        try:
            message = json.dumps(data).encode('utf-8')
            self.socket.sendall(len(message).to_bytes(4, 'big')) # Send message size first
            self.socket.sendall(message)
            # print(f"Sent request: {data}")
            return True
        except socket.timeout:
            print("Error: Socket timeout while sending request.")
            self.disconnect()
            return False
        except Exception as e:
            print(f"Error sending request: {e}")
            self.disconnect()
            return False

    def receive_response(self, buffer_size=4096):
        """Receives simulation results from UE."""
        if not self.is_connected:
            print("Error: Not connected to the simulator.")
            return None
        try:
            # Read message size
            raw_msglen = self.socket.recv(4)
            if not raw_msglen:
                print("Error: Connection closed while receiving size.")
                self.disconnect()
                return None
            msglen = int.from_bytes(raw_msglen, 'big')

            # Read the actual message
            data = b''
            while len(data) < msglen:
                chunk = self.socket.recv(min(msglen - len(data), buffer_size))
                if not chunk:
                    print("Error: Connection closed while receiving data.")
                    self.disconnect()
                    return None
                data += chunk

            response = json.loads(data.decode('utf-8'))
            # print(f"Received response: {response}")
            return response
        except socket.timeout:
            print("Error: Socket timeout while waiting for response.")
            # Consider if disconnect is appropriate here, maybe UE is just slow
            return None # Indicate timeout
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response from simulator.")
            return None
        except Exception as e:
            print(f"Error receiving response: {e}")
            self.disconnect()
            return None

    def run_simulation(self, welding_params):
        """
        Sends welding parameters to UE, runs simulation, and gets results.

        Args:
            welding_params (dict): Dictionary containing parameters like
                                   'current', 'voltage', 'speed', 'torch_angle', etc.

        Returns:
            dict: A dictionary containing simulation results (e.g.,
                  {'predicted_bead_shape': [...], 'quality_score': 0.85})
                  or None if simulation failed.
        """
        if not self.is_connected:
            if not self.connect(): # Try to reconnect if not connected
                return None # Failed to connect

        request_data = {'action': 'simulate_weld', 'parameters': welding_params}

        if self.send_request(request_data):
            response = self.receive_response()
            if response and response.get('status') == 'success':
                return response.get('results')
            elif response:
                print(f"Simulation failed on UE side: {response.get('message', 'No details')}")
                return None
            else:
                # Error handled in receive_response or send_request
                return None
        else:
            return None # Error handled in send_request

    def get_sim2real_ark_situation(self):
        """Placeholder to get Sim2Real Ark situation representation from UE."""
        # This would involve sending a specific request to UE
        print("Requesting Sim2Real Ark Situation from UE (Placeholder)")
        # response = self.send_request({'action': 'get_ark_situation'}) ...
        # return processed_response
        return {"ark_stability": 0.9, "spatter_level": 0.1} # Dummy data


if __name__ == '__main__':
    # Example Usage (requires a running simulator server counterpart)
    simulator = UnrealSimulatorInterface()

    # --- Test Connection ---
    # if simulator.connect():
    #     # --- Test Simulation Run ---
    #     test_params = {
    #         'current': 150,
    #         'voltage': 22,
    #         'speed': 300,
    #         'torch_angle': 10,
    #         'ctwd': 15
    #     }
    #     results = simulator.run_simulation(test_params)
    #     if results:
    #         print("\nSimulation Results:")
    #         print(results)
    #     else:
    #         print("\nSimulation failed or communication error.")

    #     # --- Test Disconnection ---
    #     simulator.disconnect()
    # else:
    #     print("\nFailed to connect to the simulator.")

    # --- Offline Placeholder Demo ---
    print("\n--- Running Offline Placeholder Demo ---")
    def dummy_run_simulation(params):
        print(f"Simulating weld with params: {params} (Offline Demo)")
        # Simulate some variation based on input
        score = 0.5 + (params['current'] / 400) + (params['speed'] / 2000) - (params['torch_angle']/100)
        score = max(0, min(1, score)) # Clamp between 0 and 1
        shape_options = ["Good Bead", "Undercut", "Lack of Fusion"]
        shape = shape_options[hash(str(params)) % len(shape_options)]
        time.sleep(0.1) # Simulate processing time
        return {'predicted_bead_shape': shape, 'quality_score': round(score, 3)}

    test_params = {'current': 160, 'voltage': 23, 'speed': 250, 'torch_angle': 5, 'ctwd': 12}
    dummy_results = dummy_run_simulation(test_params)
    print("Dummy Simulation Results:", dummy_results)
    