# src/ai_model.py
# Description: (MODIFIED) Defines the PyTorch AI model architecture,
#              supporting multimodal inputs (sensor, simulation features, optional images).

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import config to access model architecture parameters
# Assuming config.py is in the same src directory or python path is set
from src import config

class WeldingAIModel(nn.Module):
    """
    Advanced AI model that can combine sensor data, simulation-derived features,
    and optionally image data for welding quality prediction.
    """
    def __init__(self,
                 sensor_input_size=config.SENSOR_FEATURE_SIZE,
                 sim_feature_input_size=getattr(config, 'SIMULATION_FEATURE_SIZE', 0), # Get from config, default 0 if not present
                 use_images=config.MODEL_USES_IMAGES, # Controlled by config
                 image_input_channels=config.IMAGE_INPUT_SHAPE[0] if config.MODEL_USES_IMAGES else 1, # Default if not used
                 image_input_shape=config.IMAGE_INPUT_SHAPE if config.MODEL_USES_IMAGES else (1,1,1), # Default if not used
                 output_size=config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1,
                 # Hidden layer dimensions (can also be part of config)
                 sensor_hidden_dims=getattr(config, 'MODEL_SENSOR_HIDDEN_DIMS', [128, 64]),
                 sim_hidden_dims=getattr(config, 'MODEL_SIM_HIDDEN_DIMS', [64, 32]),
                 image_cnn_channels=getattr(config, 'MODEL_IMAGE_CNN_CHANNELS', [16, 32]), # Output channels for CNN layers
                 image_fc_hidden_dims=getattr(config, 'MODEL_IMAGE_FC_HIDDEN_DIMS', [128, 64]),
                 combined_fc_hidden_dims=getattr(config, 'MODEL_COMBINED_FC_HIDDEN_DIMS', [128, 64]),
                 dropout_rate=getattr(config, 'MODEL_DROPOUT_RATE', 0.3)
                 ):
        super().__init__()

        self.use_sensor_branch = sensor_input_size > 0
        self.use_sim_feature_branch = sim_feature_input_size > 0
        self.use_image_branch = use_images

        logger = config.logging.getLogger(__name__) # Use logger from config if available
        if not hasattr(config, 'logging'): # Fallback basic logger
            logging.basicConfig(level=config.LOG_LEVEL if hasattr(config,'LOG_LEVEL') else logging.INFO)
            logger = logging.getLogger(__name__)

        logger.info("Initializing WeldingAIModel (Advanced)...")
        logger.info(f"  Sensor Branch: {'Enabled' if self.use_sensor_branch else 'Disabled'} (Input Size: {sensor_input_size})")
        logger.info(f"  Simulation Feature Branch: {'Enabled' if self.use_sim_feature_branch else 'Disabled'} (Input Size: {sim_feature_input_size})")
        logger.info(f"  Image Branch: {'Enabled' if self.use_image_branch else 'Disabled'} (Input Shape: {image_input_shape if self.use_image_branch else 'N/A'})")
        logger.info(f"  Output Size: {output_size} (Task: {config.AI_MODEL_TYPE})")


        # --- Sensor Data Branch (MLP) ---
        if self.use_sensor_branch:
            sensor_layers = []
            current_dim = sensor_input_size
            for h_dim in sensor_hidden_dims:
                sensor_layers.append(nn.Linear(current_dim, h_dim))
                sensor_layers.append(nn.ReLU())
                sensor_layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.sensor_branch = nn.Sequential(*sensor_layers)
            self.sensor_output_dim = current_dim
        else:
            self.sensor_output_dim = 0


        # --- Simulation Feature Branch (MLP) ---
        if self.use_sim_feature_branch:
            sim_layers = []
            current_dim = sim_feature_input_size
            for h_dim in sim_hidden_dims:
                sim_layers.append(nn.Linear(current_dim, h_dim))
                sim_layers.append(nn.ReLU())
                sim_layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.sim_feature_branch = nn.Sequential(*sim_layers)
            self.sim_output_dim = current_dim
        else:
            self.sim_output_dim = 0


        # --- Image Data Branch (CNN + MLP) ---
        if self.use_image_branch:
            cnn_layers = []
            current_channels = image_input_channels
            # Example CNN: Two Conv+Pool layers
            # Layer 1
            cnn_layers.append(nn.Conv2d(current_channels, image_cnn_channels[0], kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # Halves H, W
            current_channels = image_cnn_channels[0]
            # Layer 2
            cnn_layers.append(nn.Conv2d(current_channels, image_cnn_channels[1], kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # Halves H, W again
            current_channels = image_cnn_channels[1]
            self.image_cnn_branch = nn.Sequential(*cnn_layers)

            # Calculate flattened size after convolutions and pooling
            # IMPORTANT: This calculation assumes input image_input_shape[1] (H) and image_input_shape[2] (W)
            # are divisible by 4 (due to two MaxPool2d layers with stride=2).
            # A more robust way is to pass a dummy tensor through self.image_cnn_branch
            # in __init__ to get the output shape.
            # Or use nn.AdaptiveAvgPool2d((target_H, target_W)) before flatten to get fixed size.
            if image_input_shape[1] % 4 != 0 or image_input_shape[2] % 4 != 0:
                 logger.warning(f"Image dimensions {image_input_shape[1]}x{image_input_shape[2]} might not be perfectly divisible by 4 from MaxPool layers. Flattened size calculation might be inexact.")
            flattened_h = image_input_shape[1] // 4
            flattened_w = image_input_shape[2] // 4
            image_flattened_size = current_channels * flattened_h * flattened_w
            logger.info(f"  Image CNN output channels: {current_channels}, H: {flattened_h}, W: {flattened_w}. Flattened size: {image_flattened_size}")


            image_fc_layers = []
            current_dim = image_flattened_size
            for h_dim in image_fc_hidden_dims:
                image_fc_layers.append(nn.Linear(current_dim, h_dim))
                image_fc_layers.append(nn.ReLU())
                image_fc_layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            self.image_fc_branch = nn.Sequential(*image_fc_layers)
            self.image_output_dim = current_dim
        else:
            self.image_output_dim = 0


        # --- Combined Feature Processing Branch ---
        # Sum of output dimensions from all active branches
        combined_input_dim = self.sensor_output_dim + self.sim_output_dim + self.image_output_dim
        if combined_input_dim == 0:
             raise ValueError("At least one input branch (sensor, sim_feature, or image) must be enabled and have non-zero input size.")
        logger.info(f"  Combined input dimension for final FC layers: {combined_input_dim}")

        combined_fc_layers = []
        current_dim = combined_input_dim
        for h_dim in combined_fc_hidden_dims:
            combined_fc_layers.append(nn.Linear(current_dim, h_dim))
            combined_fc_layers.append(nn.ReLU())
            combined_fc_layers.append(nn.Dropout(dropout_rate)) # Dropout after ReLU is common
            current_dim = h_dim
        self.combined_fc_branch = nn.Sequential(*combined_fc_layers)

        # --- Output Layer ---
        # output_size is 1 for regression, num_classes for classification
        self.output_layer = nn.Linear(current_dim, output_size)

        logger.info(f"WeldingAIModel (Advanced) initialization complete. Final output layer dim: {current_dim} -> {output_size}")


    def forward(self, sensor_data=None, sim_feature_data=None, image_data=None):
        """
        Forward pass of the model.

        Args:
            sensor_data (torch.Tensor, optional): Tensor of sensor data.
            sim_feature_data (torch.Tensor, optional): Tensor of simulation-derived features.
            image_data (torch.Tensor, optional): Tensor of image data (batch_size, C, H, W).

        Returns:
            torch.Tensor: Model output (logits for classification, score for regression).
        """
        active_branch_outputs = []

        # Process Sensor Data Branch
        if self.use_sensor_branch:
            if sensor_data is None:
                raise ValueError("Sensor data is expected but not provided.")
            x_sensor = self.sensor_branch(sensor_data)
            active_branch_outputs.append(x_sensor)

        # Process Simulation Feature Branch
        if self.use_sim_feature_branch:
            if sim_feature_data is None:
                raise ValueError("Simulation feature data is expected but not provided.")
            x_sim = self.sim_feature_branch(sim_feature_data)
            active_branch_outputs.append(x_sim)

        # Process Image Data Branch
        if self.use_image_branch:
            if image_data is None:
                raise ValueError("Image data is expected but not provided.")
            x_image_cnn = self.image_cnn_branch(image_data)
            x_image_flat = torch.flatten(x_image_cnn, 1) # Flatten all dimensions except batch
            x_image_fc = self.image_fc_branch(x_image_flat)
            active_branch_outputs.append(x_image_fc)


        # --- Combine Features from Active Branches ---
        # If only one branch is active, combined_features is just its output.
        # If multiple, concatenate them.
        if not active_branch_outputs:
             raise ValueError("No active input branches processed. This should not happen if __init__ validated inputs.")

        if len(active_branch_outputs) > 1:
            # Concatenate features along the feature dimension (dim=1)
            # logger.debug(f"Shapes before concat: {[feat.shape for feat in active_branch_outputs]}") # For debugging
            combined_features = torch.cat(active_branch_outputs, dim=1)
        else:
            combined_features = active_branch_outputs[0] # Only one active branch

        # Pass through combined FC layers
        x_combined_fc = self.combined_fc_branch(combined_features)

        # Final output layer
        output = self.output_layer(x_combined_fc)

        # For classification, raw logits are returned. Softmax is typically applied
        # as part of the loss function (nn.CrossEntropyLoss) or explicitly after.
        # For regression, the direct output is usually the predicted score.
        return output


if __name__ == '__main__':
    # Example Usage (Ensure config.py has the necessary SENSOR_FEATURE_SIZE, etc.)
    logger = config.logging.getLogger(__name__)
    logger.info("--- WeldingAIModel (Advanced) Example Usage ---")

    # --- Scenario 1: Sensor data only ---
    logger.info("\n--- Testing Model (Sensor Only) ---")
    # Temporarily modify config for this test if needed, or ensure config is set for sensor-only
    # For a clean test, you might create a temporary config object or mock config
    # For now, assume config is set for sensor_input_size > 0, and others are 0 or MODEL_USES_IMAGES=False
    # This requires sensor_input_size to be correctly set in config.py,
    # and MODEL_SIM_FEATURE_SIZE=0 (or sim_feature_input_size=0 passed to constructor),
    # and MODEL_USES_IMAGES=False (or use_images=False passed).
    # To test specific configurations, you'd pass specific args to WeldingAIModel constructor.

    # Example for Sensor-only:
    try:
        model_sensor_only = WeldingAIModel(
            sensor_input_size=config.SENSOR_FEATURE_SIZE, # Must be > 0
            sim_feature_input_size=0,  # Disable sim branch for this test
            use_images=False           # Disable image branch for this test
        )
        batch_size_test = getattr(config, 'BATCH_SIZE', 2) # Use a small batch for testing
        dummy_sensor_input = torch.randn(batch_size_test, config.SENSOR_FEATURE_SIZE)
        output_sensor_only = model_sensor_only(sensor_data=dummy_sensor_input)
        logger.info(f"Sensor-only model output shape: {output_sensor_only.shape}") # (batch_size, output_size)
        assert output_sensor_only.shape[0] == batch_size_test
        assert output_sensor_only.shape[1] == (config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1)
    except Exception as e:
        logger.error(f"Error in Sensor-only test: {e}", exc_info=True)


    # --- Scenario 2: Sensor + Simulation Features ---
    if hasattr(config, 'SIMULATION_FEATURE_SIZE') and config.SIMULATION_FEATURE_SIZE > 0:
        logger.info("\n--- Testing Model (Sensor + Simulation Features) ---")
        try:
            model_sensor_sim = WeldingAIModel(
                sensor_input_size=config.SENSOR_FEATURE_SIZE,
                sim_feature_input_size=config.SIMULATION_FEATURE_SIZE, # Must be > 0
                use_images=False
            )
            batch_size_test = getattr(config, 'BATCH_SIZE', 2)
            dummy_sensor_input = torch.randn(batch_size_test, config.SENSOR_FEATURE_SIZE)
            dummy_sim_input = torch.randn(batch_size_test, config.SIMULATION_FEATURE_SIZE)
            output_sensor_sim = model_sensor_sim(sensor_data=dummy_sensor_input, sim_feature_data=dummy_sim_input)
            logger.info(f"Sensor+Sim model output shape: {output_sensor_sim.shape}")
            assert output_sensor_sim.shape[0] == batch_size_test
            assert output_sensor_sim.shape[1] == (config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1)
        except Exception as e:
            logger.error(f"Error in Sensor+Sim test: {e}", exc_info=True)
    else:
        logger.info("\nSkipping Sensor + Simulation Features test as SIMULATION_FEATURE_SIZE is 0 or not defined.")


    # --- Scenario 3: Sensor + Simulation Features + Images ---
    if config.MODEL_USES_IMAGES and hasattr(config, 'SIMULATION_FEATURE_SIZE') and config.SIMULATION_FEATURE_SIZE > 0:
        logger.info("\n--- Testing Model (Sensor + Sim + Image) ---")
        try:
            model_all_inputs = WeldingAIModel(
                sensor_input_size=config.SENSOR_FEATURE_SIZE,
                sim_feature_input_size=config.SIMULATION_FEATURE_SIZE,
                use_images=True, # Enable image branch
                image_input_channels=config.IMAGE_INPUT_SHAPE[0],
                image_input_shape=config.IMAGE_INPUT_SHAPE
            )
            batch_size_test = getattr(config, 'BATCH_SIZE', 2)
            dummy_sensor_input = torch.randn(batch_size_test, config.SENSOR_FEATURE_SIZE)
            dummy_sim_input = torch.randn(batch_size_test, config.SIMULATION_FEATURE_SIZE)
            dummy_image_input = torch.randn(batch_size_test, *config.IMAGE_INPUT_SHAPE) # (batch, C, H, W)

            output_all_inputs = model_all_inputs(
                sensor_data=dummy_sensor_input,
                sim_feature_data=dummy_sim_input,
                image_data=dummy_image_input
            )
            logger.info(f"Sensor+Sim+Image model output shape: {output_all_inputs.shape}")
            assert output_all_inputs.shape[0] == batch_size_test
            assert output_all_inputs.shape[1] == (config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1)
        except Exception as e:
            logger.error(f"Error in Sensor+Sim+Image test: {e}", exc_info=True)

    elif config.MODEL_USES_IMAGES:
        logger.info("\nSkipping Sensor + Sim + Image test as SIMULATION_FEATURE_SIZE might be 0 or not defined, but MODEL_USES_IMAGES is True.")
    else:
        logger.info("\nSkipping Sensor + Sim + Image test as MODEL_USES_IMAGES is False.")

    logger.info("--- WeldingAIModel (Advanced) Example Usage Finished ---")