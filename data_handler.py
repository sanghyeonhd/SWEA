# data_handler.py
# Description: Handles data loading, preprocessing, and dataset creation.

import pandas as pd
import numpy as np
import os
# import cv2 # Uncomment if using OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader

import config

# --- Dummy Data Generation (for demonstration) ---
def generate_dummy_data(num_samples=1000):
    """Generates dummy sensor and label data if files don't exist."""
    if not os.path.exists(config.SENSOR_DATA_PATH):
        print(f"Generating dummy sensor data at {config.SENSOR_DATA_PATH}")
        sensor_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=num_samples, freq='S'),
            'current': np.random.uniform(*config.PARAM_RANGES['current'], num_samples),
            'voltage': np.random.uniform(*config.PARAM_RANGES['voltage'], num_samples),
            'speed': np.random.uniform(*config.PARAM_RANGES['speed'], num_samples),
            'temperature': np.random.normal(300, 50, num_samples) # Example sensor
        })
        sensor_data.to_csv(config.SENSOR_DATA_PATH, index=False)

    if not os.path.exists(config.LABEL_DATA_PATH):
        print(f"Generating dummy label data at {config.LABEL_DATA_PATH}")
        labels = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=num_samples, freq='S'),
            'bead_quality_class': np.random.randint(0, config.OUTPUT_CLASSES, num_samples),
            'score': np.random.rand(num_samples)
            # Add image file references if needed
            # 'image_file': [f'img_{i:04d}.png' for i in range(num_samples)]
        })
        labels.to_csv(config.LABEL_DATA_PATH, index=False)

    # Placeholder for image data directory
    if not os.path.exists(config.IMAGE_DATA_DIR):
        os.makedirs(config.IMAGE_DATA_DIR)
        print(f"Created dummy image directory: {config.IMAGE_DATA_DIR}")
        # In a real scenario, you would populate this directory with actual images

# --- Data Loading ---
def load_data():
    """Loads sensor and label data."""
    generate_dummy_data() # Generate dummy data if needed
    try:
        sensor_df = pd.read_csv(config.SENSOR_DATA_PATH, parse_dates=['timestamp'])
        label_df = pd.read_csv(config.LABEL_DATA_PATH, parse_dates=['timestamp'])
        # Merge sensor and label data based on timestamp or another key
        data_df = pd.merge(sensor_df, label_df, on='timestamp', how='inner')
        print(f"Loaded and merged data: {len(data_df)} samples.")
        return data_df
    except FileNotFoundError:
        print(f"Error: Data files not found. Please check paths in config.py")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Preprocessing ---
def preprocess_data(df):
    """Preprocesses the loaded data (scaling, normalization)."""
    if df is None:
        return None, None

    # Example: Select features and target
    sensor_features = ['current', 'voltage', 'speed', 'temperature']
    target_variable = 'bead_quality_class' # Or 'score', depending on the task

    X = df[sensor_features].values
    y = df[target_variable].values

    # Scale sensor features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data preprocessing complete (scaling).")
    # In a real scenario, add image loading and preprocessing here
    # e.g., load images listed in df, resize, normalize pixels

    return X_scaled, y, scaler # Return scaler to inverse transform later if needed

# --- PyTorch Dataset ---
class WeldingDataset(Dataset):
    """Custom Dataset for PyTorch."""
    def __init__(self, sensor_data, labels, image_dir=None, image_files=None):
        """
        Args:
            sensor_data (np.array): Preprocessed sensor data.
            labels (np.array): Corresponding labels.
            image_dir (str, optional): Directory with all images.
            image_files (list, optional): List of image filenames corresponding to samples.
        """
        self.sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # Use float if predicting score
        self.image_dir = image_dir
        self.image_files = image_files
        # Add image transforms if needed (e.g., using torchvision.transforms)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'sensors': self.sensor_data[idx], 'label': self.labels[idx]}

        if self.image_dir and self.image_files:
            # Load and preprocess image (Placeholder)
            # img_path = os.path.join(self.image_dir, self.image_files[idx])
            # try:
            #     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Example
            #     # Apply necessary transforms (resize, normalize, to tensor)
            #     image = cv2.resize(image, (config.IMAGE_INPUT_SHAPE[1], config.IMAGE_INPUT_SHAPE[2]))
            #     image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0 # Normalize
            #     sample['image'] = image
            # except Exception as e:
            #     print(f"Warning: Could not load image {img_path}: {e}")
            #     # Return a dummy tensor or handle appropriately
            #     sample['image'] = torch.zeros(config.IMAGE_INPUT_SHAPE) # Dummy image
            # For now, return only sensor data
            pass

        return sample


def get_dataloaders(test_size=0.2, validation_size=0.1):
    """Creates train, validation, and test dataloaders."""
    df = load_data()
    if df is None:
        return None, None, None

    # Handle potential image file references here if using images
    # image_files = df['image_file'].tolist() if 'image_file' in df.columns else None

    X_scaled, y, _ = preprocess_data(df)
    if X_scaled is None:
         return None, None, None

    # Split data (Train+Val / Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y if config.OUTPUT_CLASSES > 1 else None
    )

    # Split data (Train / Val)
    relative_val_size = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val if config.OUTPUT_CLASSES > 1 else None
    )

    # Create Datasets (adjust arguments if using images)
    train_dataset = WeldingDataset(X_train, y_train)
    val_dataset = WeldingDataset(X_val, y_val)
    test_dataset = WeldingDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"DataLoaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage:
    train_loader, val_loader, test_loader = get_dataloaders()
    if train_loader:
        print("\nSample batch from train_loader:")
        sample_batch = next(iter(train_loader))
        print("Sensor data shape:", sample_batch['sensors'].shape)
        print("Labels shape:", sample_batch['label'].shape)
        # print("Image data shape:", sample_batch['image'].shape) # If images are included
        