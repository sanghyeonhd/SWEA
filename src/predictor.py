# predictor.py
# Description: Uses AI model or physics simulation to predict welding outcomes.

import torch
import numpy as np
import os

from src import config
from ai_model import WeldingAIModel
from physics_interface import UnrealSimulatorInterface # Assuming this class exists

class Predictor:
    """Handles prediction using either AI or Physics Simulation."""

    def __init__(self, use_ai=True, use_physics=False):
        self.use_ai = use_ai
        self.use_physics = use_physics
        self.ai_model = None
        self.physics_simulator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None # Add scaler if needed for AI input preprocessing

        if self.use_ai:
            self._load_ai_model()
            # You might need to load the scaler used during training
            # self.scaler = load_scaler_object() # Placeholder

        if self.use_physics:
            self._initialize_physics_simulator()

    def _load_ai_model(self):
        """Loads the trained PyTorch AI model."""
        model_path = config.MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            print(f"Warning: AI model file not found at {model_path}. AI prediction unavailable.")
            self.use_ai = False
            return

        try:
            # Ensure the model architecture matches the saved state_dict
            self.ai_model = WeldingAIModel(
                sensor_input_size=config.SENSOR_FEATURE_SIZE,
                # Adjust image params if model was trained with images
                use_images=False, # Set to True if the saved model uses images
                num_classes=config.OUTPUT_CLASSES
            ).to(self.device)
            self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.ai_model.eval() # Set to evaluation mode
            print(f"AI model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            self.ai_model = None
            self.use_ai = False

    def _initialize_physics_simulator(self):
        """Initializes the interface to the physics simulator."""
        print("Initializing Physics Simulator Interface...")
        self.physics_simulator = UnrealSimulatorInterface(
            host=config.UE_SIMULATOR_IP,
            port=config.UE_SIMULATOR_PORT
        )
        # Optional: Attempt initial connection
        # if not self.physics_simulator.connect():
        #     print("Warning: Failed to connect to physics simulator initially.")
            # Decide if self.use_physics should be False if connection fails

    def predict_with_ai(self, sensor_input, image_input=None):
        """
        Predicts welding outcome using the AI model.

        Args:
            sensor_input (np.array): Numpy array of sensor values (e.g., [current, voltage, speed, temp]).
                                     Should be shape (1, num_features).
            image_input (np.array, optional): Preprocessed image data.

        Returns:
            dict: Prediction results (e.g., {'predicted_class': 1, 'probabilities': [0.1, 0.7, 0.1, 0.1]})
                  or None if prediction fails.
        """
        if not self.use_ai or self.ai_model is None:
            print("AI model not available for prediction.")
            return None

        # --- Preprocessing ---
        # Apply the same scaling as used in training
        # Example: if self.scaler: sensor_input = self.scaler.transform(sensor_input)
        # Ensure input is in the correct format (tensor, device)
        try:
            sensor_tensor = torch.tensor(sensor_input, dtype=torch.float32).to(self.device)
            if sensor_tensor.ndim == 1: # Ensure batch dimension
                 sensor_tensor = sensor_tensor.unsqueeze(0)

            image_tensor = None
            if image_input is not None and self.ai_model.use_images:
                 # Preprocess and convert image_input to tensor
                 # image_tensor = preprocess_image(image_input).to(self.device) # Placeholder
                 pass # Add image tensor creation logic

            # --- Prediction ---
            with torch.no_grad():
                # output = self.ai_model(sensor_data=sensor_tensor, image_data=image_tensor)
                output = self.ai_model(sensor_data=sensor_tensor) # Sensor only

            # --- Post-processing ---
            if config.OUTPUT_CLASSES > 1: # Classification
                probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                predicted_class = np.argmax(probabilities)
                return {'predicted_class': predicted_class.item(), 'probabilities': probabilities.tolist()}
            else: # Regression
                predicted_score = output.squeeze().cpu().numpy()
                return {'predicted_score': predicted_score.item()}

        except Exception as e:
            print(f"Error during AI prediction: {e}")
            return None

    def predict_with_physics(self, welding_params):
        """
        Predicts welding outcome using the physics simulator.

        Args:
            welding_params (dict): Dictionary of parameters for the simulator.

        Returns:
            dict: Simulation results (e.g., {'predicted_bead_shape': ..., 'quality_score': ...})
                  or None if simulation fails.
        """
        if not self.use_physics or self.physics_simulator is None:
            print("Physics simulator not available for prediction.")
            return None

        try:
            results = self.physics_simulator.run_simulation(welding_params)
            return results # Returns dict from simulator or None
        except Exception as e:
            print(f"Error during physics simulation prediction: {e}")
            return None

    def predict(self, sensor_input=None, image_input=None, welding_params=None, prefer='ai'):
        """
        Makes a prediction using the preferred method.

        Args:
            sensor_input (np.array, optional): Sensor data for AI.
            image_input (np.array, optional): Image data for AI.
            welding_params (dict, optional): Parameters for physics simulation.
            prefer (str): 'ai' or 'physics'. Method to try first.

        Returns:
            tuple: (method_used, prediction_result) or (None, None) if failed.
        """
        if prefer == 'ai' and self.use_ai and sensor_input is not None:
            ai_result = self.predict_with_ai(sensor_input, image_input)
            if ai_result is not None:
                return 'ai', ai_result
            elif self.use_physics and welding_params is not None: # Fallback to physics
                print("AI prediction failed or unavailable, falling back to physics.")
                physics_result = self.predict_with_physics(welding_params)
                if physics_result is not None:
                    return 'physics', physics_result

        elif prefer == 'physics' and self.use_physics and welding_params is not None:
            physics_result = self.predict_with_physics(welding_params)
            if physics_result is not None:
                return 'physics', physics_result
            elif self.use_ai and sensor_input is not None: # Fallback to AI
                print("Physics prediction failed or unavailable, falling back to AI.")
                ai_result = self.predict_with_ai(sensor_input, image_input)
                if ai_result is not None:
                    return 'ai', ai_result

        # If preferred method wasn't specified or failed, try the other if available
        if self.use_ai and sensor_input is not None and prefer != 'ai':
             ai_result = self.predict_with_ai(sensor_input, image_input)
             if ai_result is not None:
                 return 'ai', ai_result
        if self.use_physics and welding_params is not None and prefer != 'physics':
             physics_result = self.predict_with_physics(welding_params)
             if physics_result is not None:
                 return 'physics', physics_result

        print("Prediction failed: No suitable method available or input data missing.")
        return None, None


if __name__ == '__main__':
    # Example Usage
    # Note: This requires a trained model file (models/welding_model.pth) for AI
    # and potentially a running physics simulator for the physics part.

    # --- Initialize Predictor ---
    # Try AI first, then physics as fallback (if available)
    predictor = Predictor(use_ai=True, use_physics=True)

    # --- Example Input Data ---
    # Use realistic values within your trained data range
    # This should ideally come from a live sensor or a test dataset row
    # Remember to scale it if your AI model expects scaled input!
    dummy_sensor_data = np.array([[150, 22, 300, 350]]) # Example: current, voltage, speed, temp
    # scaled_sensor_data = predictor.scaler.transform(dummy_sensor_data) # If scaler is loaded

    dummy_welding_params = {
        'current': 150, 'voltage': 22, 'speed': 300,
        'torch_angle': 10, 'ctwd': 15, 'gas_flow': 20
    }
    # dummy_image = np.random.rand(128, 128) # Example image data (if needed)

    # --- Make Prediction ---
    print("\n--- Attempting Prediction (Prefer AI) ---")
    method, result = predictor.predict(
        sensor_input=dummy_sensor_data, # Use scaled_sensor_data if applicable
        # image_input=dummy_image, # Add if AI model uses images
        welding_params=dummy_welding_params,
        prefer='ai'
    )

    if result:
        print(f"\nPrediction successful using: {method}")
        print(f"Result: {result}")
    else:
        print("\nPrediction failed.")

    # --- Example: Predict only with Physics ---
    # print("\n--- Attempting Prediction (Physics Only) ---")
    # predictor_phys = Predictor(use_ai=False, use_physics=True)
    # method_p, result_p = predictor_phys.predict(welding_params=dummy_welding_params, prefer='physics')
    # if result_p:
    #     print(f"\nPrediction successful using: {method_p}")
    #     print(f"Result: {result_p}")
    # else:
    #     print("\nPhysics prediction failed.")

    # Close physics connection if open
    if predictor.physics_simulator and predictor.physics_simulator.is_connected:
        predictor.physics_simulator.disconnect()
    # if predictor_phys.physics_simulator and predictor_phys.physics_simulator.is_connected:
    #     predictor_phys.physics_simulator.disconnect()
    