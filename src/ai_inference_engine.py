# ai_inference_engine.py
# Description: Handles real-time prediction using the trained AI model.
# Receives data from sensor_data_handler and outputs to evaluator/logger.

import torch
import numpy as np
import os
import joblib # Recommended for saving/loading scikit-learn scalers

import config
from ai_model import WeldingAIModel
# Assume sensor_data_handler provides data (this module will consume it)
# Assume evaluator and data_logger_db will consume the output

class AIInferenceEngine:
    """
    Handles real-time AI model inference based on incoming sensor data.
    Loads a pre-trained model and scaler.
    """

    def __init__(self):
        self.ai_model = None
        self.scaler = None # Scaler object used during training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.scaler_loaded = False

        print(f"AI Inference Engine initializing on device: {self.device}")

        # Load the trained model and scaler upon initialization
        self._load_model()
        self._load_scaler()

    def _load_model(self):
        """Loads the trained PyTorch AI model from the specified path."""
        model_path = config.MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            print(f"Error: AI model file not found at {model_path}. Cannot perform AI inference.")
            self.model_loaded = False
            return

        try:
            # Re-create the model architecture. Ensure this matches the saved model!
            # NOTE: If the saved model uses images, config.MODEL_USES_IMAGES or similar
            # would be needed, and use_images=True passed here.
            # Assuming for now the model architecture is fixed or derivable from config.
            self.ai_model = WeldingAIModel(
                sensor_input_size=config.SENSOR_FEATURE_SIZE,
                num_classes=config.OUTPUT_CLASSES,
                # Assuming model does NOT use images based on predictor.py example,
                # but this should ideally be stored/loaded with the model or specified in config.
                use_images=False # TODO: Make this dynamic based on saved model/config
            ).to(self.device)

            # Load the saved state dictionary
            self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
            self.ai_model.eval()

            self.model_loaded = True
            print(f"AI model loaded successfully from {model_path}")

        except Exception as e:
            print(f"Error loading AI model from {model_path}: {e}")
            self.model_loaded = False
            self.ai_model = None # Ensure model is None if loading fails

    # TODO: Define SCALER_SAVE_PATH in config.py
    # TODO: Ensure trainer.py saves the scaler object to this path
    def _load_scaler(self):
        """Loads the trained scaler object used for input preprocessing."""
        # Assuming the scaler is saved using joblib or pickle
        scaler_path = getattr(config, 'SCALER_SAVE_PATH', 'models/scaler.pkl') # Default path

        if not os.path.exists(scaler_path):
            print(f"Warning: Scaler file not found at {scaler_path}. AI input scaling will be skipped!")
            self.scaler_loaded = False
            return

        try:
            # Load the scaler object (e.g., MinMaxScaler, StandardScaler)
            self.scaler = joblib.load(scaler_path) # Or use pickle.load
            self.scaler_loaded = True
            print(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
            self.scaler_loaded = False
            self.scaler = None # Ensure scaler is None if loading fails

    def process_sensor_data(self, raw_sensor_input):
        """
        Processes a single batch or sample of raw sensor data and performs inference.

        Args:
            raw_sensor_input (np.array or list): Raw sensor data.
                                                 Expected shape: (num_features,) for a single sample
                                                 or (batch_size, num_features) for a batch.

        Returns:
            dict or None: Prediction result dictionary (e.g., {'predicted_class': 1, 'probabilities': [...]})
                          or None if prediction fails or model is not loaded.
        """
        if not self.model_loaded:
            print("Cannot process data: AI model is not loaded.")
            return None

        # Convert input to numpy array and ensure correct shape (add batch dimension if needed)
        sensor_data_np = np.array(raw_sensor_input)
        if sensor_data_np.ndim == 1:
            sensor_data_np = sensor_data_np.reshape(1, -1) # Ensure shape (1, num_features)

        # Validate input size
        if sensor_data_np.shape[-1] != config.SENSOR_FEATURE_SIZE:
             print(f"Error: Input sensor feature size mismatch. Expected {config.SENSOR_FEATURE_SIZE}, got {sensor_data_np.shape[-1]}.")
             return None
        if sensor_data_np.ndim != 2:
             print(f"Error: Input sensor data shape must be 1D or 2D. Got {sensor_data_np.ndim}D.")
             return None


        # --- Preprocessing: Scaling ---
        # Apply the loaded scaler if available
        if self.scaler_loaded and self.scaler:
            try:
                scaled_sensor_data = self.scaler.transform(sensor_data_np)
            except Exception as e:
                 print(f"Error applying scaler: {e}")
                 return None # Scaling failed
        else:
            # Proceed without scaling, but warn if scaler was expected
            if getattr(config, 'SCALER_SAVE_PATH', 'models/scaler.pkl') != 'models/scaler.pkl': # If a custom path was specified
                 print("Warning: Scaler not loaded, processing with raw sensor data.")
            scaled_sensor_data = sensor_data_np # Use original data if scaler not loaded

        # Convert to PyTorch tensor and move to device
        try:
            sensor_tensor = torch.tensor(scaled_sensor_data, dtype=torch.float32).to(self.device)
        except Exception as e:
             print(f"Error converting data to tensor: {e}")
             return None


        # --- Inference ---
        with torch.no_grad(): # Inference mode
            try:
                # Assuming the model expects only sensor_data if use_images=False
                outputs = self.ai_model(sensor_data=sensor_tensor)
            except Exception as e:
                print(f"Error during model inference: {e}")
                return None


        # --- Post-processing ---
        # Convert output tensor to numpy and process based on task (classification/regression)
        outputs_np = outputs.cpu().numpy()

        prediction_result = {}
        if config.OUTPUT_CLASSES > 1: # Classification
            # Apply Softmax to get probabilities if the model doesn't already
            # Check model's last layer/training criterion to be sure
            # Assuming raw logits are output and CrossEntropyLoss is used in trainer
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)
            prediction_result = {
                'predicted_class': predicted_class.item(),
                'probabilities': probabilities.tolist(),
                'confidence': float(probabilities[predicted_class]) # Confidence in the predicted class
            }
        else: # Regression (Assuming output size is 1)
            predicted_score = outputs.squeeze().cpu().numpy().item() # Get scalar value
            prediction_result = {
                'predicted_score': predicted_score
            }

        # print(f"Inference result: {prediction_result}") # Optional: Log result

        return prediction_result

    # --- Example method for Sim2Real feedback (Placeholder) ---
    def get_sim2real_feedback(self, raw_sensor_input):
        """
        Processes sensor data and potentially generates feedback for Sim2Real alignment.
        This is a conceptual placeholder.
        """
        # Example: Use inference results or raw data features for feedback
        prediction = self.process_sensor_data(raw_sensor_input)
        if prediction and 'confidence' in prediction:
            # Example feedback: high confidence in 'Good' class -> signals good alignment
            # Low confidence or defect prediction -> signals potential discrepancy
            feedback_score = prediction.get('confidence', 0.0) if prediction.get('predicted_class') == 0 else (1.0 - prediction.get('confidence', 0.0))
            return {"sim2real_alignment_score": feedback_score, "ai_prediction": prediction}
        else:
             return {"sim2real_alignment_score": 0.0, "ai_prediction": prediction, "message": "Could not generate S2R feedback"}


# Example Usage (requires a dummy model and scaler file to exist)
if __name__ == '__main__':
    print("--- AI Inference Engine Example ---")

    # --- Create Dummy Config (for example purposes only) ---
    # In a real run, this would be imported from config.py
    class DummyConfig:
        MODEL_SAVE_PATH = 'dummy_welding_model.pth'
        SCALER_SAVE_PATH = 'dummy_scaler.pkl' # New path
        SENSOR_FEATURE_SIZE = 4 # Based on dummy data below
        OUTPUT_CLASSES = 4 # Based on trainer.py example
        # Add IMAGE_INPUT_SHAPE if use_images is relevant
        # Add MODEL_USES_IMAGES = False # Or True

    config = DummyConfig() # Use dummy config for this example

    # --- Create Dummy Model and Scaler Files ---
    # In a real scenario, these would be created by trainer.py
    print("Creating dummy model and scaler files...")
    try:
        # Dummy Model
        dummy_model = WeldingAIModel(sensor_input_size=config.SENSOR_FEATURE_SIZE, num_classes=config.OUTPUT_CLASSES, use_images=False)
        torch.save(dummy_model.state_dict(), config.MODEL_SAVE_PATH)

        # Dummy Scaler
        dummy_scaler = MinMaxScaler()
        dummy_scaler.fit(np.random.rand(10, config.SENSOR_FEATURE_SIZE)) # Fit with some random data
        joblib.dump(dummy_scaler, config.SCALER_SAVE_PATH)

        print("Dummy files created.")

        # --- Initialize the Inference Engine ---
        inference_engine = AIInferenceEngine()

        # --- Create Dummy Real-time Sensor Input ---
        # This would normally come from sensor_data_handler.py
        # Example single sample (shape (4,))
        dummy_live_sensor_data_1 = np.array([160.5, 23.1, 285.0, 380.2])
        # Example batch of samples (shape (2, 4))
        dummy_live_sensor_data_2 = np.array([
            [155.0, 22.5, 300.0, 350.0],
            [170.0, 24.0, 270.0, 410.0]
        ])


        # --- Process Data and Get Prediction ---
        print("\nProcessing single sensor data sample...")
        prediction_1 = inference_engine.process_sensor_data(dummy_live_sensor_data_1)
        if prediction_1:
            print("Prediction Result 1:", prediction_1)

        # Note: Current process_sensor_data is designed for single samples (reshapes to (1, -1)).
        # For batch processing, the reshaping logic needs adjustment.
        # Let's test with the reshaped single sample input explicitly.
        print("\nProcessing single sample (reshaped)...")
        prediction_1_reshaped = inference_engine.process_sensor_data(dummy_live_sensor_data_1.reshape(1, -1))
        if prediction_1_reshaped:
             print("Prediction Result 1 (Reshaped):", prediction_1_reshaped)


        print("\n--- Example Sim2Real Feedback ---")
        feedback_1 = inference_engine.get_sim2real_feedback(dummy_live_sensor_data_1)
        print("Sim2Real Feedback 1:", feedback_1)


    except Exception as e:
        print(f"An error occurred during the example run: {e}")

    finally:
        # --- Clean up Dummy Files ---
        print("\nCleaning up dummy files...")
        if os.path.exists(config.MODEL_SAVE_PATH):
            os.remove(config.MODEL_SAVE_PATH)
            print(f"Removed {config.MODEL_SAVE_PATH}")
        if os.path.exists(config.SCALER_SAVE_PATH):
            os.remove(config.SCALER_SAVE_PATH)
            print(f"Removed {config.SCALER_SAVE_PATH}")

    print("\n--- Example Finished ---")