# create_dummy_model_files.py
# Description: Creates dummy .pth and .pkl files for the AI model and scaler.
#              These files contain initial/dummy data, not trained results.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import torch
import joblib # For saving/loading scikit-learn scaler
import numpy as np # For dummy scaler fitting data
from sklearn.preprocessing import MinMaxScaler # Example scaler

# Assuming ai_model.py and config.py are in the 'src' directory
try:
    from src.ai_model import WeldingAIModel
    from src import config
    # Attempt to get specific config values, fall back to defaults if missing
    MODEL_SAVE_PATH = getattr(config, 'MODEL_SAVE_PATH', 'models/welding_model.pth')
    SCALER_SAVE_PATH = getattr(config, 'SCALER_SAVE_PATH', 'models/scaler.pkl')
    SENSOR_FEATURE_SIZE = getattr(config, 'SENSOR_FEATURE_SIZE', 4) # Default if not in config
    OUTPUT_CLASSES = getattr(config, 'OUTPUT_CLASSES', 4)       # Default if not in config
    # Assume a config variable indicates if the model architecture uses images
    # This must match how you plan to load the model in ai_inference_engine/predictor
    MODEL_USES_IMAGES = getattr(config, 'MODEL_USES_IMAGES', False) # Default assumption
    IMAGE_INPUT_SHAPE = getattr(config, 'IMAGE_INPUT_SHAPE', (1, 128, 128)) # Default if needed
except ImportError:
    print("Warning: Could not import src.config or src.ai_model. Using hardcoded dummy paths and sizes.")
    MODEL_SAVE_PATH = 'models/welding_model.pth'
    SCALER_SAVE_PATH = 'models/scaler.pkl'
    SENSOR_FEATURE_SIZE = 4
    OUTPUT_CLASSES = 4
    MODEL_USES_IMAGES = False
    IMAGE_INPUT_SHAPE = (1, 128, 128) # Still define shape even if not used, in case MODEL_USES_IMAGES changes
except AttributeError as e:
     print(f"Warning: Missing attribute in config.py: {e}. Using default values for relevant settings.")
     # Use defaults for specific missing attributes
     MODEL_SAVE_PATH = getattr(config, 'MODEL_SAVE_PATH', 'models/welding_model.pth')
     SCALER_SAVE_PATH = getattr(config, 'SCALER_SAVE_PATH', 'models/scaler.pkl')
     SENSOR_FEATURE_SIZE = getattr(config, 'SENSOR_FEATURE_SIZE', 4)
     OUTPUT_CLASSES = getattr(config, 'OUTPUT_CLASSES', 4)
     MODEL_USES_IMAGES = getattr(config, 'MODEL_USES_IMAGES', False)
     IMAGE_INPUT_SHAPE = getattr(config, 'IMAGE_INPUT_SHAPE', (1, 128, 128))


def create_dummy_pth(model_path, sensor_feature_size, output_classes, use_images, image_input_shape):
    """Creates a .pth file with the initial state_dict of a WeldingAIModel."""
    print(f"Attempting to create dummy model .pth file at: {model_path}")

    # Ensure the directory for the model path exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        # Initialize the model with the specified architecture
        # The parameters MUST match how the model will be loaded later (e.g., in ai_inference_engine.py)
        dummy_model = WeldingAIModel(
            sensor_input_size=sensor_feature_size,
            num_classes=output_classes,
            use_images=use_images,
            # Pass image_input_channels only if use_images is True,
            # as WeldingAIModel's __init__ expects it conditionally.
            **(
                {'image_input_channels': image_input_shape[0], 'image_input_shape': image_input_shape}
                if use_images else {}
            )
        )

        # Save the initial (randomly initialized) state dictionary
        torch.save(dummy_model.state_dict(), model_path)
        print(f"Successfully created dummy model file: {model_path}")
        return True
    except NameError:
         print("Error: WeldingAIModel class not found. Please ensure src.ai_model is accessible.")
         return False
    except Exception as e:
        print(f"Error creating dummy model file: {e}")
        return False

def create_dummy_scaler_pkl(scaler_path, sensor_feature_size):
    """Creates a .pkl file with a dummy fitted scaler object."""
    print(f"Attempting to create dummy scaler .pkl file at: {scaler_path}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    try:
        # Create a dummy scaler instance
        scaler = MinMaxScaler()

        # Fit the scaler with some dummy data (shape should match expected input)
        # The actual ranges don't matter for a dummy scaler, just the shape.
        dummy_data = np.random.rand(100, sensor_feature_size) # 100 samples, num_features size
        scaler.fit(dummy_data)

        # Save the fitted scaler object using joblib (recommended for scikit-learn objects)
        joblib.dump(scaler, scaler_path)
        print(f"Successfully created dummy scaler file: {scaler_path}")
        return True
    except NameError:
         print("Error: MinMaxScaler or joblib not found. Please ensure scikit-learn and joblib are installed.")
         return False
    except Exception as e:
        print(f"Error creating dummy scaler file: {e}")
        return False


if __name__ == '__main__':
    print("--- Creating Dummy Model and Scaler Files ---")

    # Check if files already exist
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    scaler_exists = os.path.exists(SCALER_SAVE_PATH)

    if model_exists:
        print(f"Dummy model file already exists at {MODEL_SAVE_PATH}. Skipping creation.")
    else:
        create_dummy_pth(
            model_path=MODEL_SAVE_PATH,
            sensor_feature_size=SENSOR_FEATURE_SIZE,
            output_classes=OUTPUT_CLASSES,
            use_images=MODEL_USES_IMAGES,
            image_input_shape=IMAGE_INPUT_SHAPE
        )

    if scaler_exists:
        print(f"Dummy scaler file already exists at {SCALER_SAVE_PATH}. Skipping creation.")
    else:
        create_dummy_scaler_pkl(
            scaler_path=SCALER_SAVE_PATH,
            sensor_feature_size=SENSOR_FEATURE_SIZE
        )

    print("--- Dummy File Creation Process Finished ---")
    # You can now use these dummy files to test components that load models/scalers.