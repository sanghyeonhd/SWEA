# src/data_handler.py
# Description: (MODIFIED) Handles loading, preprocessing, and dataset creation
#              for AI model training, supporting large datasets and simulated data.

import pandas as pd
import numpy as np
import os
import joblib # For loading scaler objects
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Provide options
import glob # For finding multiple simulation data files

import torch
from torch.utils.data import Dataset, DataLoader

# Optional: Dask for handling larger-than-memory datasets
try:
    import dask.dataframe as dd
    dask_available = True
except ImportError:
    dask_available = False
    dd = None # Define dd as None if dask is not available

from src import config # Import the main config module
# Ensure config has SENSOR_DATA_CSV_PATH, LABEL_DATA_CSV_PATH, SIMULATED_DATA_DIR, SCALER_SAVE_PATH etc.

logger = config.logging.getLogger(__name__) # Use config's logger setup if available, else basic
if not hasattr(config, 'logging'): # Fallback if config doesn't have logging configured
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config,'LOG_LEVEL') else logging.INFO)


# --- Data Loading ---
def load_historical_data(use_dask_if_available=True):
    """
    Loads historical sensor and label data from CSV files.
    Optionally uses Dask for large files if available and enabled.
    """
    sensor_path = config.SENSOR_DATA_CSV_PATH
    label_path = config.LABEL_DATA_CSV_PATH

    if not (os.path.exists(sensor_path) and os.path.exists(label_path)):
        logger.error(f"Historical sensor data ({sensor_path}) or label data ({label_path}) not found.")
        # For dummy data generation, one might call a separate script or function here.
        # generate_dummy_data_script.main() # Example
        # For this handler, assume data exists or generation is handled elsewhere.
        return None

    logger.info(f"Loading historical sensor data from: {sensor_path}")
    logger.info(f"Loading historical label data from: {label_path}")

    try:
        if use_dask_if_available and dask_available:
            logger.info("Using Dask to load historical data.")
            # Dask can infer types, but providing dtype hints can be beneficial for large files
            sensor_df = dd.read_csv(sensor_path, parse_dates=['timestamp'])
            label_df = dd.read_csv(label_path, parse_dates=['timestamp'])
            # Merge using Dask (can be more complex for distributed joins)
            # Assuming 'timestamp' is a common key and well-distributed for efficient merge
            data_df = dd.merge(sensor_df, label_df, on='timestamp', how='inner')
            # Trigger computation and convert to pandas DataFrame for further processing
            # For very large datasets, try to keep it as Dask DataFrame as long as possible.
            # This example converts to pandas, which might be a bottleneck for huge data.
            logger.info("Computing Dask DataFrame to Pandas DataFrame...")
            data_df = data_df.compute()
            logger.info("Dask computation complete.")
        else:
            logger.info("Using Pandas to load historical data.")
            sensor_df = pd.read_csv(sensor_path, parse_dates=['timestamp'])
            label_df = pd.read_csv(label_path, parse_dates=['timestamp'])
            data_df = pd.merge(sensor_df, label_df, on='timestamp', how='inner')

        logger.info(f"Loaded and merged historical data: {len(data_df)} samples.")
        if data_df.empty:
            logger.warning("Merged historical data is empty. Check timestamp alignment or data content.")
        return data_df

    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        logger.error(f"Error: Data files not found. Please check paths in config.py")
        return None
    except Exception as e:
        logger.error(f"Error loading or merging historical data: {e}", exc_info=True)
        return None


def load_simulated_data(directory_path=config.SIMULATED_DATA_DIR, file_pattern="sim_data_*.csv", use_dask_if_available=True):
    """
    Loads and concatenates simulated data from multiple CSV files in a directory.
    Each CSV should have a consistent structure, including simulation parameters and results.
    Optionally uses Dask for large files.
    """
    if not os.path.exists(directory_path):
        logger.warning(f"Simulated data directory not found: {directory_path}. No simulated data will be loaded.")
        return None

    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    if not file_paths:
        logger.warning(f"No simulated data files found matching pattern '{file_pattern}' in {directory_path}.")
        return None

    logger.info(f"Found {len(file_paths)} simulated data files to load.")
    all_sim_data = []

    try:
        if use_dask_if_available and dask_available:
            logger.info("Using Dask to load simulated data files.")
            # Dask can read multiple files directly
            # Assuming all files have consistent dtypes. Specify dtypes if known.
            sim_ddf = dd.read_csv(file_paths) # Dask automatically handles multiple files
            logger.info("Computing Dask DataFrame for simulated data...")
            sim_df_combined = sim_ddf.compute()
            logger.info("Dask computation for simulated data complete.")
        else:
            logger.info("Using Pandas to load and concatenate simulated data files.")
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                all_sim_data.append(df)
            if not all_sim_data: return None
            sim_df_combined = pd.concat(all_sim_data, ignore_index=True)

        logger.info(f"Loaded and combined simulated data: {len(sim_df_combined)} samples.")
        if sim_df_combined.empty:
            logger.warning("Combined simulated data is empty.")
        return sim_df_combined

    except Exception as e:
        logger.error(f"Error loading or combining simulated data: {e}", exc_info=True)
        return None


# --- Feature Engineering ---
def engineer_features(df, is_simulated_data=False):
    """
    Performs feature engineering on the DataFrame.
    Adds new features or modifies existing ones.
    This function should be customized based on domain knowledge.
    """
    if df is None or df.empty:
        return df

    logger.info(f"Starting feature engineering for {'simulated' if is_simulated_data else 'historical'} data...")
    df_engineered = df.copy()

    # Example 1: Calculate Heat Input (Conceptual - needs correct formula and units)
    # Assuming 'current', 'voltage', 'speed' are present and in appropriate units
    # Heat Input (kJ/mm) = (Current * Voltage * Efficiency_Factor * 60) / (Speed_mm_min * 1000)
    # Efficiency_Factor for GMAW is typically 0.8-0.9
    efficiency_factor = 0.85
    if 'current' in df_engineered.columns and 'voltage' in df_engineered.columns and 'speed' in df_engineered.columns:
        # Ensure speed is not zero to avoid division by zero
        # Ensure all values are numeric
        numeric_cols = ['current', 'voltage', 'speed']
        for col in numeric_cols:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
        df_engineered.dropna(subset=numeric_cols, inplace=True) # Drop rows where conversion failed

        if not df_engineered.empty:
             df_engineered['heat_input_calc'] = (df_engineered['current'] * df_engineered['voltage'] * efficiency_factor * 60) / \
                                             (df_engineered['speed'].replace(0, np.nan) * 1000) # Avoid division by zero
             df_engineered['heat_input_calc'].fillna(0, inplace=True) # Fill NaN from division by zero with 0 or appropriate value
             logger.info("Calculated 'heat_input_calc' feature.")


    # Example 2: Extract features from simulated data (if specific columns exist)
    if is_simulated_data:
        if 'sim_bead_width' in df_engineered.columns and 'sim_bead_height' in df_engineered.columns:
            df_engineered['sim_bead_aspect_ratio'] = df_engineered['sim_bead_width'] / df_engineered['sim_bead_height'].replace(0, np.nan)
            df_engineered['sim_bead_aspect_ratio'].fillna(0, inplace=True)
            logger.info("Calculated 'sim_bead_aspect_ratio' for simulated data.")
        # Add more extraction of meaningful features from raw simulation outputs

    # Example 3: Time-based features (if 'timestamp' exists and is datetime)
    if 'timestamp' in df_engineered.columns and pd.api.types.is_datetime64_any_dtype(df_engineered['timestamp']):
        df_engineered['hour_of_day'] = df_engineered['timestamp'].dt.hour
        df_engineered['day_of_week'] = df_engineered['timestamp'].dt.dayofweek
        logger.info("Extracted 'hour_of_day' and 'day_of_week' features.")

    logger.info("Feature engineering complete.")
    return df_engineered


# --- Preprocessing ---
def preprocess_data(df, scaler_type='minmax', scaler_path=config.SCALER_SAVE_PATH, fit_scaler=False):
    """
    Preprocesses the data: selects features, handles missing values, scales features.

    Args:
        df (pd.DataFrame): DataFrame containing combined (sensor, label, sim_features) data.
        scaler_type (str): 'minmax' or 'standard'.
        scaler_path (str): Path to load/save the scaler object.
        fit_scaler (bool): If True, fits a new scaler and saves it. Otherwise, loads an existing one.

    Returns:
        Tuple: (X_processed, y, image_paths, scaler_object) or (None, None, None, None) if error.
               X_processed contains sensor_features and sim_features.
    """
    if df is None or df.empty:
        logger.error("Preprocessing called with empty or None DataFrame.")
        return None, None, None, None

    logger.info(f"Starting data preprocessing. Fit scaler: {fit_scaler}, Scaler type: {scaler_type}")

    # --- Define Features and Target ---
    # These MUST align with the AI model's expected input structure.
    # Example: Base sensor features (ensure these exist after feature engineering)
    base_sensor_features = ['current', 'voltage', 'speed', 'temperature']
    # Example: Engineered features (if created)
    engineered_sensor_features = ['heat_input_calc', 'hour_of_day'] # Add 'day_of_week' if used
    # Example: Simulated features (if available and to be used by AI)
    simulated_features_for_ai = ['sim_bead_width', 'sim_bead_height', 'sim_bead_aspect_ratio'] # Names from simulated data CSVs

    # Select available features from the DataFrame
    # All features that will be scaled and passed to the AI model
    numerical_features_to_scale = []
    for f_list in [base_sensor_features, engineered_sensor_features, simulated_features_for_ai]:
        for f in f_list:
            if f in df.columns:
                numerical_features_to_scale.append(f)
            else:
                logger.warning(f"Feature '{f}' defined for scaling not found in DataFrame columns. It will be skipped.")
    numerical_features_to_scale = sorted(list(set(numerical_features_to_scale))) # Unique and sorted for consistency

    if not numerical_features_to_scale:
        logger.error("No numerical features selected for scaling. Cannot proceed with AI input preparation.")
        return None, None, None, None
    logger.info(f"Selected numerical features for AI input: {numerical_features_to_scale}")


    # --- Define Target Variable ---
    # Based on config.AI_MODEL_TYPE ('classification' or 'regression')
    if config.AI_MODEL_TYPE == 'classification':
        target_variable = 'bead_quality_class' # Expects integer class labels
        target_dtype = torch.long
    elif config.AI_MODEL_TYPE == 'regression':
        target_variable = 'score' # Expects float scores
        target_dtype = torch.float32
    else:
        logger.error(f"Unsupported AI_MODEL_TYPE: {config.AI_MODEL_TYPE}. Must be 'classification' or 'regression'.")
        return None, None, None, None

    if target_variable not in df.columns:
        logger.error(f"Target variable '{target_variable}' not found in DataFrame.")
        return None, None, None, None
    logger.info(f"Selected target variable: {target_variable} (for {config.AI_MODEL_TYPE})")


    # --- Handle Missing Values (Example: Simple Mean Imputation or Drop) ---
    # For numerical features to be scaled
    df_processed = df[numerical_features_to_scale + [target_variable]].copy() # Work with a copy
    for col in numerical_features_to_scale:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median() # Using median is often more robust to outliers than mean
            df_processed[col].fillna(median_val, inplace=True)
            logger.warning(f"Missing values in '{col}' filled with median ({median_val:.2f}).")
    # For target variable, usually drop rows with missing target or handle based on strategy
    df_processed.dropna(subset=[target_variable], inplace=True)
    if df_processed.empty:
         logger.error("DataFrame became empty after handling missing target values.")
         return None, None, None, None


    X_data = df_processed[numerical_features_to_scale].values
    y_data = df_processed[target_variable].values


    # --- Feature Scaling ---
    scaler = None
    if fit_scaler:
        logger.info(f"Fitting new {scaler_type} scaler.")
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type '{scaler_type}'. Using MinMaxScaler as default.")
            scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_data)
        # Save the fitted scaler
        try:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler fitted and saved to {scaler_path}")
        except Exception as e:
            logger.error(f"Error saving scaler to {scaler_path}: {e}")
            # Continue with scaled data but warn that scaler wasn't saved
    else: # Load existing scaler
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X_data)
                logger.info(f"Scaler loaded from {scaler_path} and applied.")
            except Exception as e:
                logger.error(f"Error loading or applying scaler from {scaler_path}: {e}. Proceeding without scaling numerical features.")
                X_scaled = X_data # Use unscaled data if scaler fails
                scaler = None # Ensure scaler is None
        else:
            logger.warning(f"Scaler file not found at {scaler_path}. Numerical features will not be scaled.")
            X_scaled = X_data # Use unscaled data
            scaler = None


    # --- Image Data (Placeholder - needs image paths in DataFrame) ---
    image_paths = None
    if config.MODEL_USES_IMAGES and 'image_file_path' in df.columns:
        image_paths = df['image_file_path'].tolist() # Assuming 'image_file_path' column exists
        logger.info(f"Extracted {len(image_paths)} image paths for dataset.")
    elif config.MODEL_USES_IMAGES:
        logger.warning("'MODEL_USES_IMAGES' is True, but 'image_file_path' column not found in DataFrame.")


    logger.info("Data preprocessing (feature selection, missing value handling, scaling) complete.")
    return X_scaled, y_data, image_paths, scaler, target_dtype


# --- PyTorch Dataset ---
class WeldingDataset(Dataset):
    """
    Custom Dataset for PyTorch. Handles numerical features and optional image data.
    """
    def __init__(self, numerical_data, labels, target_dtype, image_paths=None, image_transform=None):
        """
        Args:
            numerical_data (np.array): Preprocessed numerical sensor/simulation data.
            labels (np.array): Corresponding labels.
            target_dtype (torch.dtype): Dtype for the labels (e.g., torch.long for classification).
            image_paths (list, optional): List of image file paths.
            image_transform (callable, optional): torchvision.transforms to apply to images.
        """
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=target_dtype)
        self.image_paths = image_paths
        self.image_transform = image_transform

        # Basic validation
        if len(self.numerical_data) != len(self.labels):
             raise ValueError("Mismatch in length of numerical_data and labels.")
        if self.image_paths and (len(self.image_paths) != len(self.labels)):
             raise ValueError("Mismatch in length of image_paths and labels.")

        logger.debug(f"WeldingDataset initialized. Numerical data shape: {self.numerical_data.shape}, Labels shape: {self.labels.shape}")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'numerical_features': self.numerical_data[idx],
            'label': self.labels[idx]
        }

        if self.image_paths:
            try:
                # Example image loading (requires PIL/OpenCV and torchvision.transforms)
                # from PIL import Image
                # image = Image.open(self.image_paths[idx]).convert('RGB') # Or 'L' for grayscale
                # if self.image_transform:
                #     image = self.image_transform(image)
                # sample['image_features'] = image
                # Placeholder for actual image loading
                sample['image_path'] = self.image_paths[idx] # Pass path for now
                sample['image_features'] = torch.randn(config.IMAGE_INPUT_SHAPE) # Dummy image tensor if actual loading not implemented
                # logger.debug(f"Image path for sample {idx}: {self.image_paths[idx]}")

            except Exception as e:
                logger.error(f"Error loading image {self.image_paths[idx]} for sample {idx}: {e}")
                # Handle missing/corrupt image: return dummy, skip sample, or raise error
                # For this example, we'll return a dummy tensor.
                sample['image_features'] = torch.zeros(config.IMAGE_INPUT_SHAPE)
                sample['image_path'] = self.image_paths[idx] # Still pass the path

        return sample


def get_dataloaders(test_size=0.2, validation_size=0.1, batch_size=None, fit_scaler_on_train=True):
    """
    Loads all data (historical, simulated), preprocesses it, splits into
    train/validation/test sets, and creates PyTorch DataLoaders.
    """
    logger.info("Starting DataLoader creation process...")
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # 1. Load Historical Data
    df_historical = load_historical_data()
    if df_historical is None or df_historical.empty:
        logger.error("Failed to load historical data. Cannot create DataLoaders.")
        return None, None, None

    # 2. Load Simulated Data (Optional)
    df_simulated = load_simulated_data()
    # Combine if simulated data is available
    # This requires careful alignment of columns and data types
    # and potentially a flag to distinguish simulated vs real data if model needs it.
    if df_simulated is not None and not df_simulated.empty:
        # --- Data Alignment & Merging Strategy for Sim/Real ---
        # Option A: Concatenate directly if schemas align (requires common target variable).
        # Option B: Use simulated data for pre-training or as separate features.
        # Option C: Merge based on simulated input params matching real input params (complex).
        # This example assumes simulated data has *some* overlap or comparable features/target
        # for a simple concatenation. A real strategy needs careful thought.
        # Ensure columns used for X and y exist in both and are compatible.
        # For simplicity, let's assume concatenation and feature engineering handles differences.
        try:
            # Before concatenating, ensure common columns (used for features/target) have consistent types.
            # This is a placeholder for more robust schema alignment.
            df_combined = pd.concat([df_historical, df_simulated], ignore_index=True, sort=False)
            # Fill NaNs that might arise from non-overlapping columns if not handling them earlier
            # df_combined.fillna(value=..., inplace=True) # Or specific strategies
            logger.info(f"Combined historical and simulated data. Total samples: {len(df_combined)}")
        except Exception as e_concat:
            logger.error(f"Error concatenating historical and simulated data: {e_concat}. Using historical data only.")
            df_combined = df_historical
    else:
        df_combined = df_historical


    # 3. Feature Engineering
    df_engineered = engineer_features(df_combined, is_simulated_data=(df_simulated is not None))
    if df_engineered is None or df_engineered.empty:
        logger.error("Feature engineering resulted in empty DataFrame. Cannot create DataLoaders.")
        return None, None, None

    # 4. Split Data (Train / Validation / Test) - Before scaling for proper scaler fitting
    # Stratify by target if it's a classification task and target is available
    target_var = 'bead_quality_class' if config.AI_MODEL_TYPE == 'classification' else 'score'
    stratify_col = df_engineered[target_var] if config.AI_MODEL_TYPE == 'classification' and target_var in df_engineered else None

    logger.info(f"Splitting data into train/validation/test sets. Total samples: {len(df_engineered)}")
    if len(df_engineered) < 10: # Basic check for enough data
        logger.error(f"Not enough data samples ({len(df_engineered)}) for train/val/test split. Need at least ~10.")
        return None, None, None


    # Split into Train+Validation and Test
    df_train_val, df_test = train_test_split(
        df_engineered,
        test_size=test_size,
        random_state=42,
        stratify=stratify_col
    )

    # Split Train+Validation into Train and Validation
    # Adjust stratify for the second split
    stratify_col_train_val = df_train_val[target_var] if config.AI_MODEL_TYPE == 'classification' and target_var in df_train_val else None
    relative_val_size = validation_size / (1.0 - test_size) # Validation size relative to train_val set
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=42,
        stratify=stratify_col_train_val
    )
    logger.info(f"Data split: Train={len(df_train)}, Validation={len(df_val)}, Test={len(df_test)}")

    # 5. Preprocess each split (fit scaler on train, transform val/test)
    # The `preprocess_data` function handles feature selection, missing values, and scaling.
    X_train, y_train, img_train, scaler, target_dtype_train = preprocess_data(df_train, fit_scaler=fit_scaler_on_train)
    if X_train is None:
        logger.error("Preprocessing failed for training data.")
        return None, None, None
    # Important: Use the scaler fitted on training data to transform validation and test data
    X_val, y_val, img_val, _, target_dtype_val = preprocess_data(df_val, scaler_path=config.SCALER_SAVE_PATH, fit_scaler=False)
    X_test, y_test, img_test, _, target_dtype_test = preprocess_data(df_test, scaler_path=config.SCALER_SAVE_PATH, fit_scaler=False)

    if X_val is None or X_test is None:
        logger.error("Preprocessing failed for validation or test data.")
        return None, None, None

    # Ensure target_dtype is consistent (should be, as it's derived from config.AI_MODEL_TYPE)
    target_dtype = target_dtype_train


    # 6. Create PyTorch Datasets
    # Define image transformations if using images (example)
    # image_transforms = None
    # if config.MODEL_USES_IMAGES:
    #     from torchvision import transforms
    #     image_transforms = transforms.Compose([
    #         transforms.Resize((config.IMAGE_INPUT_SHAPE[1], config.IMAGE_INPUT_SHAPE[2])),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # If RGB
    #     ])

    logger.info("Creating PyTorch Datasets...")
    train_dataset = WeldingDataset(X_train, y_train, target_dtype, image_paths=img_train, image_transform=None) # Pass actual transforms
    val_dataset = WeldingDataset(X_val, y_val, target_dtype, image_paths=img_val, image_transform=None)
    test_dataset = WeldingDataset(X_test, y_test, target_dtype, image_paths=img_test, image_transform=None)

    # 7. Create DataLoaders
    # Consider adding num_workers for faster data loading in parallel if not using Dask for this part
    # Pin memory if using GPU: pin_memory=torch.cuda.is_available()
    num_workers = getattr(config, 'DATALOADER_NUM_WORKERS', 0) # Default to 0 (main process)
    pin_memory = getattr(config, 'DATALOADER_PIN_MEMORY', torch.cuda.is_available())

    logger.info(f"Creating DataLoaders with Batch Size: {batch_size}, Num Workers: {num_workers}, Pin Memory: {pin_memory}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logger.info("--- Data Handler (Advanced) Example ---")

    # --- Ensure dummy data and config for testing ---
    # This assumes create_dummy_sensor_data_csv.py and create_dummy_labels_csv.py
    # have been run, or that real data paths are correctly set in config.py.
    # Also assumes a scaler might exist or will be created.

    if not (os.path.exists(config.SENSOR_DATA_CSV_PATH) and os.path.exists(config.LABEL_DATA_CSV_PATH)):
        logger.error("Dummy/Real historical data files not found. Please create them first.")
        logger.error(f"Expected sensor data at: {config.SENSOR_DATA_CSV_PATH}")
        logger.error(f"Expected label data at: {config.LABEL_DATA_CSV_PATH}")
    else:
        logger.info("Attempting to create DataLoaders...")
        # Fit scaler on train set first time, then use existing for subsequent runs if needed
        # To force re-fitting scaler, delete the scaler.pkl file or set fit_scaler_on_train=True
        train_loader, val_loader, test_loader = get_dataloaders(fit_scaler_on_train=True)

        if train_loader and val_loader and test_loader:
            logger.info("DataLoaders created successfully.")

            # Example: Iterate over a few batches from train_loader
            logger.info("\nSample batches from train_loader:")
            for i, batch in enumerate(train_loader):
                if i >= 2: # Show first 2 batches
                    break
                logger.info(f" Batch {i+1}:")
                logger.info(f"  Numerical Features Shape: {batch['numerical_features'].shape}")
                logger.info(f"  Labels Shape: {batch['label'].shape}")
                if 'image_features' in batch: # If images are included and loaded
                     logger.info(f"  Image Features Shape: {batch['image_features'].shape}")
                elif 'image_path' in batch: # If only paths are passed
                     logger.info(f"  First Image Path in Batch: {batch['image_path'][0] if isinstance(batch['image_path'], list) else batch['image_path']}")


            # Test loading data without fitting scaler (assuming scaler.pkl exists from previous run)
            logger.info("\nAttempting to create DataLoaders using existing scaler...")
            # Ensure scaler file exists for this test
            if not os.path.exists(config.SCALER_SAVE_PATH):
                 logger.warning(f"Scaler file {config.SCALER_SAVE_PATH} not found. Cannot test loading with existing scaler. Please run with fit_scaler_on_train=True first.")
            else:
                train_loader_no_fit, _, _ = get_dataloaders(fit_scaler_on_train=False)
                if train_loader_no_fit:
                    logger.info("DataLoaders created successfully using existing scaler.")
        else:
            logger.error("Failed to create DataLoaders in the example.")

    logger.info("--- Data Handler (Advanced) Example Finished ---")