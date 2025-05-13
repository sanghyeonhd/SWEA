# create_dummy_sensor_data_csv.py
# Description: Generates a dummy sensor_data.csv file with realistic-looking data.
#              Matches the structure and data types expected by data_handler.py.

import pandas as pd
import numpy as np
import os
import datetime # For timestamp generation
import random # For potential variations beyond ranges

# Assuming config.py is in the 'src' directory
try:
    from src import config
    SENSOR_DATA_PATH = getattr(config, 'SENSOR_DATA_PATH', 'data/dummy_sensor_data.csv')
    PARAM_RANGES = getattr(config, 'PARAM_RANGES', {
        'current': (80, 200),
        'voltage': (15, 30),
        'speed': (100, 500)
    })
    SENSOR_FEATURE_SIZE = getattr(config, 'SENSOR_FEATURE_SIZE', 4) # Default to 4 features
    # Use start time consistent with dummy label data if possible
    DUMMY_START_TIME = datetime.datetime.fromisoformat(getattr(config, 'DUMMY_START_TIME', '2024-01-01T00:00:00+00:00')) # Example, might need config
except ImportError:
    print("Warning: Could not import src.config. Using hardcoded dummy path and values.")
    SENSOR_DATA_PATH = 'data/dummy_sensor_data.csv'
    PARAM_RANGES = {
        'current': (80, 200),
        'voltage': (15, 30),
        'speed': (100, 500)
    }
    SENSOR_FEATURE_SIZE = 4
    DUMMY_START_TIME = datetime.datetime.fromisoformat('2024-01-01T00:00:00+00:00')
except AttributeError as e:
     print(f"Warning: Missing attribute in config.py: {e}. Using default values for relevant settings.")
     SENSOR_DATA_PATH = getattr(config, 'SENSOR_DATA_PATH', 'data/dummy_sensor_data.csv')
     PARAM_RANGES = getattr(config, 'PARAM_RANGES', { # Use default if missing
        'current': (80, 200), 'voltage': (15, 30), 'speed': (100, 500)
     })
     SENSOR_FEATURE_SIZE = getattr(config, 'SENSOR_FEATURE_SIZE', 4)
     DUMMY_START_TIME = datetime.datetime.fromisoformat(getattr(config, 'DUMMY_START_TIME', '2024-01-01T00:00:00+00:00'))


# Define the expected sensor feature names based on SENSOR_FEATURE_SIZE
# This list MUST match the features expected by data_handler.preprocess_data
# and the order expected by ai_model.WeldingAIModel if SENSOR_FEATURE_SIZE=4
EXPECTED_SENSOR_FEATURES = ['current', 'voltage', 'speed', 'temperature']

# Ensure the number of defined features matches SENSOR_FEATURE_SIZE
if len(EXPECTED_SENSOR_FEATURES) != SENSOR_FEATURE_SIZE:
    print(f"Error: Defined EXPECTED_SENSOR_FEATURES ({len(EXPECTED_SENSOR_FEATURES)}) does not match config.SENSOR_FEATURE_SIZE ({SENSOR_FEATURE_SIZE}).")
    print("Please update either the list or the config value.")
    # You might want to exit here in a real script if this is a fatal mismatch
    # For this example, we will proceed but the generated data might not match expectations.
    # Adjust the list to match the size by truncating or adding placeholders if needed.
    if len(EXPECTED_SENSOR_FEATURES) < SENSOR_FEATURE_SIZE:
        while len(EXPECTED_SENSOR_FEATURES) < SENSOR_FEATURE_SIZE:
            EXPECTED_SENSOR_FEATURES.append(f'sensor_{len(EXPECTED_SENSOR_FEATURES)}')
        print(f"Adjusted feature list to match size: {EXPECTED_SENSOR_FEATURES}")
    elif len(EXPECTED_SENSOR_FEATURES) > SENSOR_FEATURE_SIZE:
        EXPECTED_SENSOR_FEATURES = EXPECTED_SENSOR_FEATURES[:SENSOR_FEATURE_SIZE]
        print(f"Truncated feature list to match size: {EXPECTED_SENSOR_FEATURES}")


def generate_dummy_sensor_data(num_samples=1000, output_path=SENSOR_DATA_PATH, sensor_features=None, param_ranges=None, start_time=DUMMY_START_TIME):
    """
    Generates and saves a dummy sensor data CSV file.

    Args:
        num_samples (int): Number of data rows to generate.
        output_path (str): The path where the CSV file will be saved.
        sensor_features (list): List of sensor names (column headers).
        param_ranges (dict): Dictionary of parameter name to (min, max) range.
        start_time (datetime): The starting timestamp for the data.
    """
    if sensor_features is None:
        sensor_features = EXPECTED_SENSOR_FEATURES
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    print(f"Generating dummy sensor data ({num_samples} samples) at: {output_path}")
    print(f"Features: {sensor_features}")

    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Generate timestamps
    # Assuming data points are 1 second apart, consistent with data_handler dummy generation
    timestamps = pd.date_range(start=start_time, periods=num_samples, freq='S')

    # 2. Generate sensor data for each feature
    data = {'timestamp': timestamps}
    for feature in sensor_features:
        if feature in param_ranges and param_ranges[feature][0] is not None and param_ranges[feature][1] is not None:
            # Use uniform distribution for parameters with defined ranges
            min_val, max_val = param_ranges[feature]
            data[feature] = np.random.uniform(min_val, max_val, num_samples)
        elif feature == 'temperature':
            # Use normal distribution for temperature (example from original data_handler)
            mean_temp = 350 # Example mean
            std_dev_temp = 50 # Example standard deviation
            data[feature] = np.random.normal(mean_temp, std_dev_temp, num_samples)
            # Ensure temperature stays somewhat realistic (e.g., non-negative)
            data[feature] = np.maximum(data[feature], 0) # Clamp at 0
        else:
            # Generate generic random data for other features
            print(f"Warning: No specific range/distribution defined for feature '{feature}'. Using random uniform [0, 1].")
            data[feature] = np.random.rand(num_samples) * 100 # Example range [0, 100]


    # Create DataFrame
    sensor_df = pd.DataFrame(data)

    # Save to CSV
    try:
        sensor_df.to_csv(output_path, index=False)
        print(f"Dummy sensor data file created successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving dummy sensor data file: {e}")
        return False


if __name__ == '__main__':
    print("--- Creating Dummy Sensor Data CSV ---")

    # Check if file already exists
    if os.path.exists(SENSOR_DATA_PATH):
        print(f"Dummy sensor data file already exists at {SENSOR_DATA_PATH}. Skipping creation.")
        print("If you want to re-create it, please delete the existing file manually.")
    else:
        # Call the generation function
        generate_dummy_sensor_data(
            num_samples=1000, # Generate 1000 rows
            output_path=SENSOR_DATA_PATH,
            sensor_features=EXPECTED_SENSOR_FEATURES,
            param_ranges=PARAM_RANGES,
            start_time=DUMMY_START_TIME
        )

    print("--- Dummy Sensor Data CSV Creation Process Finished ---")