# create_dummy_labels_csv.py
# Description: Generates a dummy labels.csv file with realistic-looking data.
#              Matches the structure expected by data_handler.py and evaluator.py.

import pandas as pd
import numpy as np
import os
import datetime # For timestamp generation

# Assuming config.py is in the 'src' directory
try:
    from src import config
    LABEL_DATA_PATH = getattr(config, 'LABEL_DATA_PATH', 'data/dummy_labels.csv')
    OUTPUT_CLASSES = getattr(config, 'OUTPUT_CLASSES', 4) # Default to 4 classes if not in config
    # Use start time consistent with dummy sensor data if possible
    DUMMY_START_TIME = datetime.datetime.fromisoformat(getattr(config, 'DUMMY_START_TIME', '2024-01-01T00:00:00+00:00')) # Example, might need config
except ImportError:
    print("Warning: Could not import src.config. Using hardcoded dummy path and values.")
    LABEL_DATA_PATH = 'data/dummy_labels.csv'
    OUTPUT_CLASSES = 4
    DUMMY_START_TIME = datetime.datetime.fromisoformat('2024-01-01T00:00:00+00:00')
except AttributeError as e:
     print(f"Warning: Missing attribute in config.py: {e}. Using default values for relevant settings.")
     LABEL_DATA_PATH = getattr(config, 'LABEL_DATA_PATH', 'data/dummy_labels.csv')
     OUTPUT_CLASSES = getattr(config, 'OUTPUT_CLASSES', 4)
     DUMMY_START_TIME = datetime.datetime.fromisoformat(getattr(config, 'DUMMY_START_TIME', '2024-01-01T00:00:00+00:00'))


def generate_dummy_labels(num_samples=1000, output_path=LABEL_DATA_PATH, num_classes=OUTPUT_CLASSES, start_time=DUMMY_START_TIME):
    """
    Generates and saves a dummy labels CSV file.

    Args:
        num_samples (int): Number of data rows to generate.
        output_path (str): The path where the CSV file will be saved.
        num_classes (int): The number of distinct quality classes.
        start_time (datetime): The starting timestamp for the data.
    """
    print(f"Generating dummy label data ({num_samples} samples) at: {output_path}")

    # Ensure the directory for the output path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Generate timestamps
    # Assuming data points are 1 second apart, consistent with data_handler dummy generation
    timestamps = pd.date_range(start=start_time, periods=num_samples, freq='S')

    # 2. Generate bead_quality_class
    # Random integers between 0 and num_classes-1
    bead_quality_classes = np.random.randint(0, num_classes, num_samples)

    # 3. Generate score
    # Random float values between 0.0 and 1.0 (example range)
    scores = np.random.rand(num_samples)

    # Create DataFrame
    labels_df = pd.DataFrame({
        'timestamp': timestamps,
        'bead_quality_class': bead_quality_classes,
        'score': scores
    })

    # Save to CSV
    try:
        labels_df.to_csv(output_path, index=False)
        print(f"Dummy labels file created successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving dummy labels file: {e}")
        return False


if __name__ == '__main__':
    print("--- Creating Dummy Labels CSV ---")

    # Check if file already exists
    if os.path.exists(LABEL_DATA_PATH):
        print(f"Dummy labels file already exists at {LABEL_DATA_PATH}. Skipping creation.")
        print("If you want to re-create it, please delete the existing file manually.")
    else:
        # Call the generation function
        generate_dummy_labels(
            num_samples=1000,
            output_path=LABEL_DATA_PATH,
            num_classes=OUTPUT_CLASSES,
            start_time=DUMMY_START_TIME
        )

    print("--- Dummy Labels CSV Creation Process Finished ---")