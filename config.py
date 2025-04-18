# config.py
# Description: Configuration settings for the welding prediction system.

# --- Input Parameters ---
# Environmental Variables (Example ranges or typical values)
PARAM_RANGES = {
    'current': (80, 200),  # Amperes
    'voltage': (15, 30),   # Volts
    'speed': (100, 500),   # mm/min
    'gas_flow': (15, 25),  # L/min (Assuming Argon 100%)
    'heat_input': (None, None) # Calculated or specific value
}

# Adjustment Variables (Example ranges or typical values)
ADJUSTMENT_PARAMS = {
    'torch_angle': (0, 45), # Degrees
    'ctwd': (10, 20)        # Contact Tip to Work Distance (mm)
    # Add other relevant adjustment parameters
}

# --- Data ---
SENSOR_DATA_PATH = 'dummy_sensor_data.csv'
LABEL_DATA_PATH = 'dummy_labels.csv' # Contains bead shape classification/scores
IMAGE_DATA_DIR = 'data/images/' # Directory containing arc/bead images (placeholder)

# --- Physics Simulation (Unreal Engine Interface) ---
UE_SIMULATOR_IP = '127.0.0.1' # IP address of the machine running UE simulation
UE_SIMULATOR_PORT = 9999      # Port for communication

# --- AI Model (PyTorch) ---
MODEL_SAVE_PATH = 'models/welding_model.pth'
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
# Define input/output sizes based on your specific data and model architecture
SENSOR_FEATURE_SIZE = 4 # current, voltage, speed, temperature (example)
IMAGE_INPUT_SHAPE = (1, 128, 128) # Example: Grayscale image size (C, H, W)
OUTPUT_CLASSES = 4 # e.g., 0: Complete Fusion, 1: Incomplete, 2: Undercut, 3: Hot Tear

# --- Evaluation ---
QUALITY_METRICS = ['completeness_score', 'defect_probability']
