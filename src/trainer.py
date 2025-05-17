# src/trainer.py
# Description: (MODIFIED) Contains the ADVANCED training loop and evaluation logic for the AI model.
#              Supports multimodal data, distributed training (conceptual), TensorBoard logging,
#              and improved regression evaluation.

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
import os
import joblib # For saving scaler object
import time
import logging

# For Distributed Training (DDP)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# For TensorBoard Logging
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except ImportError:
    tensorboard_available = False
    SummaryWriter = None

from src import config # Import the main config module
from src.ai_model import WeldingAIModel # Import the advanced model
from src.data_handler import get_dataloaders # Import the advanced data loader getter

# Setup logging (use config's logger if available, else basic)
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') else logging.getLogger(__name__)
if not hasattr(config, 'logging'):
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.INFO)


# --- Distributed Training Setup Function (Called by run_training_ddp) ---
def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    # Initialize the process group
    # Backend can be 'nccl' (for NVIDIA GPUs), 'gloo' (for CPU/GPU), or 'mpi'
    dist.init_process_group(config.DISTRIBUTED_BACKEND, rank=rank, world_size=world_size)
    logger.info(f"DDP Initialized: Rank {rank}/{world_size} on backend {config.DISTRIBUTED_BACKEND}")

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    logger.info("DDP Cleanup complete.")


# --- Early Stopping Callback (Example) ---
class EarlyStopping:
    def __init__(self, patience=getattr(config, 'TRAINING_EARLY_STOPPING_PATIENCE', 10),
                 min_delta=getattr(config, 'TRAINING_EARLY_STOPPING_MIN_DELTA', 0.001),
                 mode='min', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'min' for loss (lower is better), 'max' for accuracy/F1 (higher is better).
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = float('inf') if mode == 'min' else float('-inf')

        if mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")

    def __call__(self, current_metric_val):
        score_improved = False
        if self.best_score is None:
            self.best_score = current_metric_val
            score_improved = True
        elif self.mode == 'min':
            if current_metric_val < self.best_score - self.min_delta:
                self.best_score = current_metric_val
                score_improved = True
        elif self.mode == 'max': # mode == 'max'
            if current_metric_val > self.best_score + self.min_delta:
                self.best_score = current_metric_val
                score_improved = True

        if score_improved:
            self.val_metric_best = self.best_score # Update for external use
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: Validation metric improved ({self.val_metric_best:.6f}). Counter reset.")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: Validation metric did not improve for {self.counter} epochs. Best: {self.val_metric_best:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"EarlyStopping: Patience of {self.patience} epochs reached. Triggering early stop.")
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num, ddp_rank=None, writer=None):
    """Runs one training epoch, supporting DDP and TensorBoard."""
    model.train() # Set model to training mode
    running_loss = 0.0
    epoch_preds = []
    epoch_labels = []
    num_samples_processed = 0

    if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler) and ddp_rank is not None:
        dataloader.sampler.set_epoch(epoch_num) # Important for shuffling with DDP

    for i, batch in enumerate(dataloader):
        # Move data to device
        sensor_features = batch.get('numerical_features').to(device) if batch.get('numerical_features') is not None else None
        sim_features = batch.get('simulation_features').to(device) if batch.get('simulation_features') is not None else None # Assuming this key from dataset
        image_features = batch.get('image_features').to(device) if batch.get('image_features') is not None else None
        labels = batch['label'].to(device)

        optimizer.zero_grad() # Zero the parameter gradients

        # Forward pass - adapt to model's expected inputs
        outputs = model(sensor_data=sensor_features,
                        sim_feature_data=sim_features, # Pass sim features if model accepts
                        image_data=image_features)     # Pass image features if model accepts

        loss = criterion(outputs, labels) # Calculate loss

        loss.backward() # Backward pass
        optimizer.step() # Optimize

        running_loss += loss.item() * labels.size(0) # loss.item() is avg loss for batch
        num_samples_processed += labels.size(0)

        # --- Store predictions and labels for epoch metrics ---
        if config.AI_MODEL_TYPE == 'classification':
             _, predicted_classes = torch.max(outputs.data, 1)
             epoch_preds.extend(predicted_classes.cpu().numpy())
             epoch_labels.extend(labels.cpu().numpy())
        elif config.AI_MODEL_TYPE == 'regression':
             epoch_preds.extend(outputs.detach().cpu().numpy().flatten()) # Store raw regression outputs
             epoch_labels.extend(labels.cpu().numpy().flatten())

        # Log batch loss (optional, can be verbose)
        if (i + 1) % getattr(config, 'TRAINING_LOG_BATCH_INTERVAL', 50) == 0:
             current_avg_loss = running_loss / num_samples_processed
             if ddp_rank is None or ddp_rank == 0: # Log only from rank 0 in DDP
                 logger.info(f'  Epoch {epoch_num+1}, Batch {i+1}/{len(dataloader)}, Avg Batch Loss: {current_avg_loss:.4f}')
                 if writer:
                     writer.add_scalar('Loss/train_batch', loss.item(), epoch_num * len(dataloader) + i)


    epoch_loss_avg = running_loss / num_samples_processed

    # --- Calculate epoch metrics ---
    # In DDP, metrics should be calculated across all processes or only on rank 0 after gathering results.
    # This example calculates on rank 0 if ddp_rank is not None.
    # For robust DDP metrics, gather predictions/labels from all ranks.
    # For simplicity, this example calculates metrics per-rank and rank 0 logs.

    metrics = {'loss': epoch_loss_avg}
    if config.AI_MODEL_TYPE == 'classification':
        # Ensure epoch_labels and epoch_preds are not empty before calculating metrics
        if epoch_labels and epoch_preds:
            metrics['accuracy'] = accuracy_score(epoch_labels, epoch_preds)
            # Note: For DDP, these metrics are per-process. For global metrics, gather first.
            # precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='weighted', zero_division=0)
            # metrics['precision'] = precision
            # metrics['recall'] = recall
            # metrics['f1'] = f1
        else:
             metrics['accuracy'] = 0.0 # Default if no data processed (e.g., single sample DDP)

    elif config.AI_MODEL_TYPE == 'regression':
        if epoch_labels and epoch_preds:
            metrics['mse'] = mean_squared_error(epoch_labels, epoch_preds)
            metrics['mae'] = mean_absolute_error(epoch_labels, epoch_preds)
            # metrics['r2'] = r2_score(epoch_labels, epoch_preds) # R2 can be tricky to interpret
        else:
            metrics['mse'] = float('inf')
            metrics['mae'] = float('inf')

    return metrics


def evaluate_model(model, dataloader, criterion, device, epoch_num, ddp_rank=None, writer=None, eval_type='Validation'):
    """Evaluates the model on the validation or test set."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    epoch_preds = []
    epoch_labels = []
    num_samples_processed = 0

    with torch.no_grad(): # Disable gradient calculation
        for batch in dataloader:
            sensor_features = batch.get('numerical_features').to(device) if batch.get('numerical_features') is not None else None
            sim_features = batch.get('simulation_features').to(device) if batch.get('simulation_features') is not None else None
            image_features = batch.get('image_features').to(device) if batch.get('image_features') is not None else None
            labels = batch['label'].to(device)

            outputs = model(sensor_data=sensor_features, sim_feature_data=sim_features, image_data=image_features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            num_samples_processed += labels.size(0)

            if config.AI_MODEL_TYPE == 'classification':
                _, predicted_classes = torch.max(outputs.data, 1)
                epoch_preds.extend(predicted_classes.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
            elif config.AI_MODEL_TYPE == 'regression':
                epoch_preds.extend(outputs.detach().cpu().numpy().flatten())
                epoch_labels.extend(labels.cpu.numpy().flatten())


    epoch_loss_avg = running_loss / num_samples_processed if num_samples_processed > 0 else 0

    metrics = {'loss': epoch_loss_avg}
    # Similar to train_epoch, DDP metrics need careful handling (gather from all ranks for global view)
    if config.AI_MODEL_TYPE == 'classification':
        if epoch_labels and epoch_preds:
            metrics['accuracy'] = accuracy_score(epoch_labels, epoch_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='weighted', zero_division=0)
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            if ddp_rank is None or ddp_rank == 0: # Log metrics from rank 0
                logger.info(f'{eval_type} Results: Loss: {epoch_loss_avg:.4f}, Acc: {metrics["accuracy"]:.4f}, F1: {f1:.4f}')
                if writer:
                    writer.add_scalar(f'Loss/{eval_type.lower()}', epoch_loss_avg, epoch_num)
                    writer.add_scalar(f'Accuracy/{eval_type.lower()}', metrics['accuracy'], epoch_num)
                    writer.add_scalar(f'F1-Score/{eval_type.lower()}', f1, epoch_num)
        else:
             metrics['accuracy'] = 0.0; metrics['f1'] = 0.0 # Defaults

    elif config.AI_MODEL_TYPE == 'regression':
        if epoch_labels and epoch_preds:
            metrics['mse'] = mean_squared_error(epoch_labels, epoch_preds)
            metrics['mae'] = mean_absolute_error(epoch_labels, epoch_preds)
            metrics['r2'] = r2_score(epoch_labels, epoch_preds)
            if ddp_rank is None or ddp_rank == 0:
                logger.info(f'{eval_type} Results: Loss(Crit): {epoch_loss_avg:.4f}, MSE: {metrics["mse"]:.4f}, MAE: {metrics["mae"]:.4f}, R2: {metrics["r2"]:.4f}')
                if writer:
                    writer.add_scalar(f'Loss/{eval_type.lower()}', epoch_loss_avg, epoch_num) # Criterion loss
                    writer.add_scalar(f'MSE/{eval_type.lower()}', metrics['mse'], epoch_num)
                    writer.add_scalar(f'MAE/{eval_type.lower()}', metrics['mae'], epoch_num)
                    writer.add_scalar(f'R2-Score/{eval_type.lower()}', metrics['r2'], epoch_num)
        else:
            metrics['mse'] = float('inf'); metrics['mae'] = float('inf'); metrics['r2'] = float('-inf') # Defaults


    return metrics


def train_worker(rank, world_size, # DDP args
                 # Other training args passed through
                 dataloaders, model_architecture_params, scaler_object_from_data_handler):
    """
    Main training worker function for DDP or single GPU/CPU training.
    """
    is_ddp = world_size > 1
    if is_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}") # Assign specific GPU for this DDP process
        torch.cuda.set_device(device)
    else: # Single GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0 # For logging and saving, treat as rank 0

    logger.info(f"[Rank {rank}] Using device: {device}")

    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

    # --- Model Initialization ---
    # Pass all necessary architecture params from config (via model_architecture_params dict)
    model = WeldingAIModel(**model_architecture_params).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False) # Set find_unused_parameters based on model

    # --- Loss and Optimizer ---
    if config.AI_MODEL_TYPE == 'classification':
        criterion = nn.CrossEntropyLoss()
        # For best model tracking, higher accuracy/F1 is better
        early_stopping_mode = 'max'
        best_val_metric_init = float('-inf')
        metric_to_track = 'accuracy' # Or 'f1'
    elif config.AI_MODEL_TYPE == 'regression':
        criterion = nn.MSELoss() # Or nn.L1Loss() for MAE
        # For best model tracking, lower MSE/MAE is better
        early_stopping_mode = 'min'
        best_val_metric_init = float('inf')
        metric_to_track = 'mse' # Or 'mae'
    else:
        raise ValueError(f"Unsupported AI_MODEL_TYPE: {config.AI_MODEL_TYPE}")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=getattr(config, 'TRAINING_WEIGHT_DECAY', 0.01))
    # Optional: Learning rate scheduler
    scheduler = None
    if getattr(config, 'TRAINING_LR_SCHEDULER_USE', False):
         scheduler_type = getattr(config, 'TRAINING_LR_SCHEDULER_TYPE', 'ReduceLROnPlateau')
         if scheduler_type == 'ReduceLROnPlateau':
              scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=early_stopping_mode,
                                                               factor=getattr(config, 'TRAINING_LR_FACTOR', 0.1),
                                                               patience=getattr(config, 'TRAINING_LR_PATIENCE', 5))
         # Add other schedulers like CosineAnnealingLR, StepLR etc.

    # --- TensorBoard Writer (only on rank 0 if DDP) ---
    writer = None
    if (rank == 0 or not is_ddp) and tensorboard_available and getattr(config, 'TENSORBOARD_LOG_DIR', None):
        log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"[Rank {rank}] TensorBoard logging to: {log_dir}")


    # --- Early Stopping Initialization ---
    early_stopper = None
    if getattr(config, 'TRAINING_USE_EARLY_STOPPING', False):
         early_stopper = EarlyStopping(mode=early_stopping_mode, verbose=(rank == 0 or not is_ddp))


    # --- Training Loop ---
    best_val_metric = best_val_metric_init
    best_epoch = -1

    if rank == 0 or not is_ddp: logger.info("\n--- Starting Training Loop ---")
    for epoch in range(config.NUM_EPOCHS):
        if rank == 0 or not is_ddp: logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, ddp_rank=rank if is_ddp else None, writer=writer)
        if rank == 0 or not is_ddp:
            log_str = f"Epoch {epoch+1} Train: Loss={train_metrics['loss']:.4f}"
            if 'accuracy' in train_metrics: log_str += f", Acc={train_metrics['accuracy']:.4f}"
            if 'mse' in train_metrics: log_str += f", MSE={train_metrics['mse']:.4f}"
            logger.info(log_str)
            if writer: writer.add_scalar(f'Loss/train_epoch', train_metrics['loss'], epoch)


        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device, epoch, ddp_rank=rank if is_ddp else None, writer=writer, eval_type='Validation')
        current_val_metric_to_track = val_metrics.get(metric_to_track, best_val_metric_init if early_stopping_mode == 'max' else -best_val_metric_init)


        # LR Scheduler step (if using ReduceLROnPlateau, step with validation metric)
        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
             scheduler.step(current_val_metric_to_track)
        elif scheduler: # For other schedulers that step per epoch
             scheduler.step()


        # Save the best model (only on rank 0 if DDP)
        if rank == 0 or not is_ddp:
            improved = False
            if early_stopping_mode == 'min': # Lower is better (e.g., MSE)
                if current_val_metric_to_track < best_val_metric:
                    best_val_metric = current_val_metric_to_track
                    improved = True
            else: # Higher is better (e.g., Accuracy, F1)
                if current_val_metric_to_track > best_val_metric:
                    best_val_metric = current_val_metric_to_track
                    improved = True

            if improved:
                best_epoch = epoch + 1
                try:
                    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
                    # If DDP, save the underlying model's state_dict
                    model_state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
                    torch.save(model_state_to_save, config.MODEL_SAVE_PATH)
                    logger.info(f"*** Best model saved at epoch {best_epoch} with Val {metric_to_track}: {best_val_metric:.4f} to {config.MODEL_SAVE_PATH} ***")
                except Exception as e_save:
                    logger.error(f"Error saving model: {e_save}")

            # Early stopping check
            if early_stopper:
                if early_stopper(current_val_metric_to_track): # Pass the metric used for comparison
                    logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                    break # Exit training loop


    # --- Training Finished ---
    if rank == 0 or not is_ddp:
        logger.info("\n--- Training Finished ---")
        logger.info(f"Best model found at epoch {best_epoch} with Validation {metric_to_track}: {best_val_metric:.4f}")
        logger.info(f"Model saved to {config.MODEL_SAVE_PATH}")

        # Save the scaler object used for training data
        if scaler_object_from_data_handler:
            try:
                os.makedirs(os.path.dirname(config.SCALER_SAVE_PATH), exist_ok=True)
                joblib.dump(scaler_object_from_data_handler, config.SCALER_SAVE_PATH)
                logger.info(f"Scaler object saved to {config.SCALER_SAVE_PATH}")
            except Exception as e_scaler:
                logger.error(f"Error saving scaler object: {e_scaler}")
        else:
            logger.warning("No scaler object provided from data_handler to save.")


        # --- Final Evaluation on Test Set (using the best saved model) ---
        if os.path.exists(config.MODEL_SAVE_PATH):
            logger.info("\n--- Evaluating Best Model on Test Set ---")
            # Load the best model state for final evaluation
            # Re-initialize model and load state_dict (important if DDP was used, as model is wrapped)
            final_model = WeldingAIModel(**model_architecture_params).to(device) # Create clean instance
            try:
                final_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
                evaluate_model(final_model, test_loader, criterion, device, config.NUM_EPOCHS, ddp_rank=None, writer=writer, eval_type='Test') # Final eval, epoch num can be total_epochs
            except Exception as e_load:
                 logger.error(f"Error loading best model for final test set evaluation: {e_load}")
        else:
             logger.warning("Best model file not found. Skipping test set evaluation.")


    if writer: # Close TensorBoard writer
        writer.close()

    if is_ddp: # Clean up DDP
        cleanup_ddp()


def run_training():
    """Main function to run the training process, handling DDP if configured."""
    world_size = getattr(config, 'WORLD_SIZE', 1) # Get world_size from config, default to 1 (no DDP)
    use_ddp = world_size > 1 and torch.cuda.is_available() and torch.cuda.device_count() >= world_size

    # --- Data Loading and Initial Scaler Fitting (Done by rank 0 or main process before DDP spawn) ---
    logger.info("Starting AI Model Training Process...")
    # Get DataLoaders (and potentially the scaler fitted on the *entire* training portion of data)
    # The get_dataloaders function from advanced data_handler now returns train, val, test loaders
    # AND the scaler object fitted on the training data split.
    logger.info("Loading data and fitting initial scaler (if applicable)...")
    # Assume data_handler.get_dataloaders() now also returns the fitted scaler from the training split
    # This scaler will be used by all DDP processes for consistency if needed (though DDP typically handles its own data splits)
    # However, for a unified scaler to be saved, rank 0 should ideally create it.
    # For simplicity, we assume get_dataloaders can provide it or it's handled by data_handler's preprocess.
    # Let's assume `get_dataloaders` fits scaler on train data if `fit_scaler_on_train=True` (default)
    # AND Importantly, the `preprocess_data` function within `get_dataloaders` needs to return the
    # scaler object so we can save it later.
    # We will pass this scaler to the train_worker for saving purposes.

    # Let's refine get_dataloaders in data_handler.py to return the scaler.
    # For now, assume data_handler.preprocess_data saves the scaler if fit_scaler=True
    # and train_worker can load it if needed, OR we pass the scaler instance.
    # The `preprocess_data` in data_handler.py was modified to return scaler.
    # Let's simulate that `get_dataloaders` might also return it for `run_training` to manage.

    # Simplified: get dataloaders. The scaler fitting happens *inside* get_dataloaders on the training split.
    # The scaler object itself needs to be saved by the main process (rank 0 in DDP).
    # `get_dataloaders` must ensure that the `scaler.pkl` is saved if `fit_scaler_on_train=True`.
    # Then, `AIInferenceEngine` can load it.
    # For this trainer to explicitly save it, data_handler needs to return it.
    # Let's assume data_handler returns: train_loader, val_loader, test_loader, fitted_training_scaler_object
    # So, `get_dataloaders` in `data_handler.py` needs modification if this is the desired flow for saving scaler here.

    # For now, we will fetch the scaler from the saved path inside train_worker if needed,
    # and rank 0 of train_worker will be responsible for saving the *training-fitted* scaler
    # that `data_handler.get_dataloaders(fit_scaler_on_train=True)` created and saved.
    # This means `data_handler.preprocess_data` (when `fit_scaler=True`) IS RESPONSIBLE FOR SAVING THE SCALER.
    # And `train_worker` only re-saves it to confirm (or could skip saving if data_handler already does it).
    # The `preprocess_data` function in the advanced `data_handler` already saves the scaler.
    # So, `scaler_object_from_data_handler` can be `None` here, and `AIInferenceEngine` will load it.
    # However, if we want trainer.py to explicitly manage and save it after all training:
    # train_loader, val_loader, test_loader, fitted_scaler = get_dataloaders_and_scaler() # Modified get_dataloaders
    # For current data_handler.py, it saves scaler inside preprocess_data.

    train_loader, val_loader, test_loader = get_dataloaders(fit_scaler_on_train=True) # This ensures scaler is fitted and saved by data_handler
    if not train_loader:
        logger.error("Failed to load data. Exiting training.")
        return

    dataloaders_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # --- Prepare Model Architecture Parameters from Config ---
    # This dictionary will be passed to WeldingAIModel constructor in each worker
    model_params = {
        'sensor_input_size': config.SENSOR_FEATURE_SIZE,
        'sim_feature_input_size': getattr(config, 'SIMULATION_FEATURE_SIZE', 0),
        'use_images': config.MODEL_USES_IMAGES,
        'image_input_channels': config.IMAGE_INPUT_SHAPE[0] if config.MODEL_USES_IMAGES else 1,
        'image_input_shape': config.IMAGE_INPUT_SHAPE if config.MODEL_USES_IMAGES else (1,1,1),
        'output_size': config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1,
        'sensor_hidden_dims': getattr(config, 'MODEL_SENSOR_HIDDEN_DIMS', [128, 64]),
        'sim_hidden_dims': getattr(config, 'MODEL_SIM_HIDDEN_DIMS', [64, 32]),
        'image_cnn_channels': getattr(config, 'MODEL_IMAGE_CNN_CHANNELS', [16, 32]),
        'image_fc_hidden_dims': getattr(config, 'MODEL_IMAGE_FC_HIDDEN_DIMS', [128, 64]),
        'combined_fc_hidden_dims': getattr(config, 'MODEL_COMBINED_FC_HIDDEN_DIMS', [128, 64]),
        'dropout_rate': getattr(config, 'MODEL_DROPOUT_RATE', 0.3)
    }

    # --- SCALER OBJECT ---
    # The scaler is fitted and saved by data_handler.preprocess_data(..., fit_scaler=True).
    # AIInferenceEngine will load it. If this trainer needs to save it again, it can load it here.
    # For simplicity, assume data_handler handles saving the definitive training scaler.
    # If train_worker needs to save it, it should load it here.
    scaler_to_save = None
    if os.path.exists(config.SCALER_SAVE_PATH):
        try:
            scaler_to_save = joblib.load(config.SCALER_SAVE_PATH)
            logger.info(f"Scaler object loaded from {config.SCALER_SAVE_PATH} to be potentially re-saved by trainer.")
        except Exception as e_load_scaler:
            logger.error(f"Could not load scaler from {config.SCALER_SAVE_PATH} for re-saving: {e_load_scaler}")


    # --- Run Training ---
    if use_ddp:
        logger.info(f"Using DistributedDataParallel (DDP) with world_size={world_size}")
        # Spawn DDP processes
        # Each process will call train_worker(rank, world_size, ...)
        mp.spawn(train_worker,
                 args=(world_size, dataloaders_dict, model_params, scaler_to_save), # Pass scaler object
                 nprocs=world_size,
                 join=True)
    else:
        logger.info("Running training on a single device (CPU or GPU if available).")
        # Call train_worker directly for single device training
        train_worker(0, 1, dataloaders_dict, model_params, scaler_to_save) # rank=0, world_size=1 for non-DDP


if __name__ == '__main__':
    run_training()