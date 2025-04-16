# ai_model.py
# Description: Defines the PyTorch AI model architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class WeldingAIModel(nn.Module):
    """
    Example AI model combining sensor and (optionally) image data.
    Adjust architecture based on actual data and requirements.
    """
    def __init__(self, sensor_input_size=config.SENSOR_FEATURE_SIZE,
                 image_input_channels=config.IMAGE_INPUT_SHAPE[0],
                 num_classes=config.OUTPUT_CLASSES,
                 use_images=False): # Set to True if image data is available and used
        super().__init__()

        self.use_images = use_images
        hidden_dim1 = 128
        hidden_dim2 = 64

        # --- Sensor Data Branch ---
        self.sensor_fc1 = nn.Linear(sensor_input_size, hidden_dim1)
        self.sensor_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.sensor_dropout = nn.Dropout(0.3)

        # --- Image Data Branch (Example CNN) ---
        if self.use_images:
            self.conv1 = nn.Conv2d(image_input_channels, 16, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves H, W
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves H, W again

            # Calculate flattened size after convolutions and pooling
            # Example: If input is 1x128x128 -> pool1 -> 1x64x64 -> pool2 -> 1x32x32
            # Flattened size = 32 * (128 // 4) * (128 // 4) = 32 * 32 * 32 = 32768
            # IMPORTANT: Adjust this calculation based on config.IMAGE_INPUT_SHAPE and layers
            image_feature_size = 32 * (config.IMAGE_INPUT_SHAPE[1] // 4) * (config.IMAGE_INPUT_SHAPE[2] // 4)
            self.image_fc1 = nn.Linear(image_feature_size, hidden_dim1)
            self.image_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.image_dropout = nn.Dropout(0.4)
            combined_feature_size = hidden_dim2 + hidden_dim2 # Sensor + Image
        else:
            combined_feature_size = hidden_dim2 # Only Sensor features

        # --- Combined Branch ---
        self.fc_combined1 = nn.Linear(combined_feature_size, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, num_classes) # For classification
        # If predicting a score (regression), change output_layer:
        # self.output_layer = nn.Linear(hidden_dim2, 1)

        print("WeldingAIModel initialized.")
        print(f"Using image data: {self.use_images}")
        print(f"Sensor input size: {sensor_input_size}")
        if self.use_images:
             print(f"Image input shape: {config.IMAGE_INPUT_SHAPE}")
        print(f"Output classes/size: {num_classes}")


    def forward(self, sensor_data, image_data=None):
        # --- Process Sensor Data ---
        x_sensor = F.relu(self.sensor_fc1(sensor_data))
        x_sensor = self.sensor_dropout(x_sensor)
        x_sensor = F.relu(self.sensor_fc2(x_sensor)) # Shape: (batch_size, hidden_dim2)

        if self.use_images and image_data is not None:
            # --- Process Image Data ---
            x_image = F.relu(self.conv1(image_data))
            x_image = self.pool1(x_image)
            x_image = F.relu(self.conv2(x_image))
            x_image = self.pool2(x_image)

            x_image = torch.flatten(x_image, 1) # Flatten all dimensions except batch

            x_image = F.relu(self.image_fc1(x_image))
            x_image = self.image_dropout(x_image)
            x_image = F.relu(self.image_fc2(x_image)) # Shape: (batch_size, hidden_dim2)

            # --- Combine Features ---
            x_combined = torch.cat((x_sensor, x_image), dim=1)

        else:
            # Use only sensor data if images are not used or not provided
            x_combined = x_sensor

        # --- Final Layers ---
        x_combined = F.relu(self.fc_combined1(x_combined))
        output = self.output_layer(x_combined)

        # If regression (predicting a score), no activation needed here usually
        # If classification, you might apply Softmax later (often combined with loss function)
        return output

if __name__ == '__main__':
    # Example usage:
    # Create dummy input data
    batch_size = config.BATCH_SIZE
    dummy_sensor_input = torch.randn(batch_size, config.SENSOR_FEATURE_SIZE)
    dummy_image_input = torch.randn(batch_size, *config.IMAGE_INPUT_SHAPE)

    # --- Model without images ---
    print("\n--- Testing Model (Sensor Only) ---")
    model_sensor_only = WeldingAIModel(use_images=False)
    output_sensor_only = model_sensor_only(sensor_data=dummy_sensor_input)
    print("Sensor-only model output shape:", output_sensor_only.shape) # (batch_size, num_classes)

    # --- Model with images ---
    # print("\n--- Testing Model (Sensor + Image) ---")
    # model_with_images = WeldingAIModel(use_images=True)
    # output_with_images = model_with_images(sensor_data=dummy_sensor_input, image_data=dummy_image_input)
    # print("Sensor+Image model output shape:", output_with_images.shape) # (batch_size, num_classes)
    