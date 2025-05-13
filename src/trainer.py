# trainer.py
# Description: Contains the training loop and evaluation logic for the AI model.

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
import os
from src import config
from ai_model import WeldingAIModel
from data_handler import get_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one training epoch."""
    model.train() # Set model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for i, batch in enumerate(dataloader):
        sensors = batch['sensors'].to(device)
        labels = batch['label'].to(device)
        # images = batch.get('image', None) # Get images if they exist in the batch
        # if images is not None:
        #     images = images.to(device)

        optimizer.zero_grad() # Zero the parameter gradients

        # Forward pass (adjust call based on whether images are used)
        # outputs = model(sensor_data=sensors, image_data=images)
        outputs = model(sensor_data=sensors) # Sensor only for now

        loss = criterion(outputs, labels) # Calculate loss

        loss.backward() # Backward pass
        optimizer.step() # Optimize

        running_loss += loss.item()

        # Store predictions and labels for epoch metrics
        # Convert outputs to predictions (e.g., class index for classification)
        if isinstance(criterion, nn.CrossEntropyLoss):
             _, predicted = torch.max(outputs.data, 1)
             all_preds.extend(predicted.cpu().numpy())
             all_labels.extend(labels.cpu().numpy())
        # Add handling for regression if needed (e.g., just store outputs)

        if (i + 1) % 50 == 0: # Print progress every 50 batches
             print(f'  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')


    epoch_loss = running_loss / len(dataloader)

    # Calculate epoch metrics (accuracy for classification example)
    if isinstance(criterion, nn.CrossEntropyLoss):
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc
    else: # Regression or other task
        # Calculate appropriate metrics (e.g., MSE) if needed
        return epoch_loss, None # Return None or other relevant metric


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on the validation or test set."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            sensors = batch['sensors'].to(device)
            labels = batch['label'].to(device)
            # images = batch.get('image', None)
            # if images is not None:
            #     images = images.to(device)

            # outputs = model(sensor_data=sensors, image_data=images)
            outputs = model(sensor_data=sensors) # Sensor only for now

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            # Add handling for regression

    avg_loss = running_loss / len(dataloader)

    # Calculate metrics
    if isinstance(criterion, nn.CrossEntropyLoss):
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        print(f'\nEvaluation Results:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1-Score: {f1:.4f}')
        return avg_loss, accuracy, f1
    else: # Regression
        # Example: Calculate Mean Squared Error
        # all_preds = torch.cat(all_preds).cpu().numpy() # If storing raw outputs
        # all_labels = torch.cat(all_labels).cpu().numpy()
        # mse = mean_squared_error(all_labels, all_preds)
        # print(f'\nEvaluation Results:')
        # print(f'  Loss (MSE): {avg_loss:.4f}') # Criterion might already be MSE
        # print(f'  MSE: {mse:.4f}')
        # return avg_loss, mse, None # Return relevant regression metrics
        print(f'\nEvaluation Results:')
        print(f'  Loss: {avg_loss:.4f}')
        return avg_loss, None, None # Placeholder


def run_training():
    """Main function to run the training process."""
    print("Starting AI Model Training Process...")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    if not train_loader:
        print("Failed to load data. Exiting training.")
        return

    # --- Model ---
    print("Initializing model...")
    # Adjust use_images=True if using image data
    model = WeldingAIModel(use_images=False, num_classes=config.OUTPUT_CLASSES).to(device)

    # --- Loss and Optimizer ---
    # Use CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()
    # If regression (predicting a score), use:
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- Training Loop ---
    best_val_metric = 0.0 # Or infinity for loss-based metrics like MSE
    best_epoch = -1

    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}" + (f", Accuracy: {train_acc:.4f}" if train_acc is not None else ""))

        # Validation phase
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}" + (f", Accuracy: {val_acc:.4f}" if val_acc is not None else ""))

        # Save the best model based on validation metric (e.g., accuracy or F1)
        current_metric = val_acc if val_acc is not None else -val_loss # Use accuracy, or negative loss if accuracy is None

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"*** Best model saved at epoch {best_epoch} with Validation Metric: {best_val_metric:.4f} ***")

    print("\n--- Training Finished ---")
    print(f"Best model found at epoch {best_epoch} with Validation Metric: {best_val_metric:.4f}")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating Best Model on Test Set ---")
    # Load the best model state
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    evaluate_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    run_training()
    