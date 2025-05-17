# src/ai_inference_engine.py
# Description: (MODIFIED) Handles real-time prediction using the trained AI model.
#              Supports ONNX Runtime for optimized inference, batch processing,
#              and consumes sensor data from a message queue.

import torch
import numpy as np
import os
import joblib # For loading scikit-learn scalers
import json
import time
import threading
import queue

# ONNX Runtime for optimized inference
try:
    import onnxruntime
    onnx_available = True
except ImportError:
    onnxruntime = None
    onnx_available = False

# Message Queue client (e.g., pika for RabbitMQ)
try:
    import pika
    pika_available = True
except ImportError:
    pika = None

from src import config # Import the main config module
from src.ai_model import WeldingAIModel # Still needed for PyTorch fallback or if ONNX conversion is done here
# from src.data_logger_db import DataLoggerDB # For logging predictions

# Setup logging (use config's logger if available, else basic)
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') else logging.getLogger(__name__)
if not hasattr(config, 'logging'):
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.INFO)


class AIInferenceEngine:
    """
    Handles real-time AI model inference using either PyTorch or ONNX Runtime.
    Consumes sensor data from a message queue and publishes predictions.
    """
    _instance = None # Singleton pattern
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_logger=None): # DataLoggerDB instance
        with self._lock:
            if self._initialized: return
            self._initialized = True

            logger.info("AI Inference Engine (Advanced) initializing...")
            self.data_logger = data_logger

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"AI Inference Engine targeting device: {self.device}")

            self.pytorch_model = None
            self.onnx_session = None
            self.scaler = None
            self.model_loaded = False
            self.scaler_loaded = False
            self.using_onnx = False

            # Load scaler first, as it's needed for both PyTorch and ONNX input prep
            self._load_scaler()

            # Load model (ONNX попытается первым, если включено и доступно)
            self.use_onnx_preference = getattr(config, 'USE_ONNX_FOR_INFERENCE', True)
            if self.use_onnx_preference and onnx_available and os.path.exists(config.ONNX_MODEL_PATH):
                self._load_onnx_model()
                if self.model_loaded: self.using_onnx = True
            
            if not self.model_loaded: # Fallback to PyTorch if ONNX failed or not preferred
                if self.use_onnx_preference:
                     logger.warning("ONNX model loading failed or not preferred/available. Attempting to load PyTorch model.")
                self._load_pytorch_model()
                self.using_onnx = False # Ensure this is false if PyTorch is loaded

            if not self.model_loaded:
                 logger.error("Failed to load any AI model (PyTorch or ONNX). AI Inference will be unavailable.")


            # --- Message Queue Setup (Subscriber for sensor data, Publisher for predictions) ---
            self._mq_connection_subscriber = None
            self._mq_channel_subscriber = None
            self._mq_connection_publisher = None # Separate connection for publishing if needed
            self._mq_channel_publisher = None
            self._mq_subscriber_thread = None
            self._stop_mq_event = threading.Event()
            self._incoming_sensor_data_queue = queue.Queue(maxsize=getattr(config, 'AI_INFERENCE_INPUT_QUEUE_SIZE', 200))
            self._inference_worker_thread = None

            if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
                if not pika:
                    logger.error("Pika library not found. MQ features for AI Inference Engine disabled.")
                else:
                    # Publisher connection (can be same as subscriber if careful with threading)
                    if self._connect_mq_publisher():
                         logger.info("AI Inference: MQ Publisher connected.")
                    else:
                         logger.error("AI Inference: Failed to connect MQ Publisher.")
                    # Subscriber connection and thread will be started by start_consuming_sensor_data()
            elif config.USE_MESSAGE_QUEUE:
                 logger.warning(f"MQ type '{config.MQ_TYPE}' not supported for AI Inference Engine yet.")


    def _load_pytorch_model(self):
        """Loads the trained PyTorch AI model."""
        model_path = config.MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            logger.warning(f"PyTorch model file not found at {model_path}.")
            return
        try:
            # Ensure architecture parameters match the saved model
            model_params = {
                'sensor_input_size': config.SENSOR_FEATURE_SIZE,
                'sim_feature_input_size': getattr(config, 'SIMULATION_FEATURE_SIZE', 0),
                'use_images': config.MODEL_USES_IMAGES,
                'image_input_channels': config.IMAGE_INPUT_SHAPE[0] if config.MODEL_USES_IMAGES else 1,
                'image_input_shape': config.IMAGE_INPUT_SHAPE if config.MODEL_USES_IMAGES else (1,1,1),
                'output_size': config.OUTPUT_CLASSES if config.AI_MODEL_TYPE == 'classification' else 1,
            } # Add other architecture params from config if WeldingAIModel __init__ expects them
            self.pytorch_model = WeldingAIModel(**model_params).to(self.device)
            self.pytorch_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.pytorch_model.eval()
            self.model_loaded = True
            logger.info(f"PyTorch AI model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading PyTorch AI model: {e}", exc_info=True)
            self.pytorch_model = None
            self.model_loaded = False


    def _load_onnx_model(self):
        """Loads an ONNX model using ONNX Runtime."""
        onnx_path = config.ONNX_MODEL_PATH
        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX model file not found at {onnx_path}.")
            return
        if not onnx_available:
            logger.warning("ONNX Runtime library not available. Cannot load ONNX model.")
            return
        try:
            # Specify providers: CUDAExecutionProvider for GPU, CPUExecutionProvider for CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            self.model_loaded = True
            logger.info(f"ONNX AI model loaded successfully from {onnx_path} using providers: {self.onnx_session.get_providers()}")
            # You might want to log input/output names for debugging:
            # logger.debug(f"ONNX Model Inputs: {[inp.name for inp in self.onnx_session.get_inputs()]}")
            # logger.debug(f"ONNX Model Outputs: {[out.name for out in self.onnx_session.get_outputs()]}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}", exc_info=True)
            self.onnx_session = None
            self.model_loaded = False


    def _load_scaler(self):
        """Loads the trained scaler object."""
        scaler_path = config.SCALER_SAVE_PATH
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found at {scaler_path}. Input scaling will be unavailable. This is critical.")
            self.scaler_loaded = False
            return
        try:
            self.scaler = joblib.load(scaler_path)
            self.scaler_loaded = True
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}", exc_info=True)
            self.scaler = None
            self.scaler_loaded = False


    def _preprocess_input_batch(self, raw_sensor_batch_np):
        """Prepares a batch of raw sensor data for inference (scaling)."""
        if not self.scaler_loaded or self.scaler is None:
            logger.warning("Scaler not loaded. Performing inference on raw (unscaled) data. This may lead to poor results.")
            return raw_sensor_batch_np # Return raw data if no scaler

        try:
            # Ensure input is 2D numpy array for scaler
            if raw_sensor_batch_np.ndim == 1:
                 raw_sensor_batch_np = raw_sensor_batch_np.reshape(1, -1)
            if raw_sensor_batch_np.shape[1] != config.SENSOR_FEATURE_SIZE: # Assuming scaler was fit on SENSOR_FEATURE_SIZE
                 logger.error(f"Scaler input feature mismatch. Expected {config.SENSOR_FEATURE_SIZE}, got {raw_sensor_batch_np.shape[1]}.")
                 return None

            scaled_batch_np = self.scaler.transform(raw_sensor_batch_np)
            return scaled_batch_np
        except Exception as e:
            logger.error(f"Error during input scaling: {e}", exc_info=True)
            return None # Or return raw_sensor_batch_np with a warning


    def _postprocess_output_batch(self, model_outputs_np):
        """Postprocesses a batch of model outputs into final prediction results."""
        # model_outputs_np is a numpy array of shape (batch_size, output_size)
        batch_predictions = []
        for i in range(model_outputs_np.shape[0]):
            single_output = model_outputs_np[i, :] # Get output for one sample
            prediction_result = {}
            if config.AI_MODEL_TYPE == 'classification':
                # Assuming output is logits, apply softmax
                probabilities = F.softmax(torch.tensor(single_output), dim=0).numpy() # Apply softmax
                predicted_class_idx = np.argmax(probabilities)
                prediction_result = {
                    'predicted_class': int(predicted_class_idx),
                    'probabilities': probabilities.tolist(),
                    'confidence': float(probabilities[predicted_class_idx])
                }
            elif config.AI_MODEL_TYPE == 'regression':
                prediction_result = {
                    'predicted_score': float(single_output[0]) # Assuming output_size is 1
                }
            batch_predictions.append(prediction_result)
        return batch_predictions


    def process_sensor_data_batch(self, raw_sensor_batch, sim_feature_batch=None, image_batch=None):
        """
        Processes a batch of raw sensor data (and optional sim/image data) and performs inference.

        Args:
            raw_sensor_batch (np.array or list of lists): Batch of raw sensor data.
                                                        Expected shape: (batch_size, num_sensor_features).
            sim_feature_batch (np.array or list of lists, optional): Batch of simulation features.
                                                        Expected shape: (batch_size, num_sim_features).
            image_batch (np.array or list of Tensors, optional): Batch of preprocessed image data.
                                                        Expected shape: (batch_size, C, H, W).
        Returns:
            list of dict or None: List of prediction result dictionaries, or None if failure.
        """
        if not self.model_loaded:
            logger.error("Cannot process data: AI model is not loaded.")
            return None
        if not self.scaler_loaded: # Critical for good predictions
             logger.error("Cannot process data accurately: Scaler is not loaded.")
             # return None # Option: refuse to predict if scaler is missing

        # --- 1. Convert to NumPy and Preprocess Sensor Data (Scaling) ---
        try:
            sensor_batch_np = np.array(raw_sensor_batch, dtype=np.float32)
            if sensor_batch_np.ndim == 1 and sensor_batch_np.shape[0] == config.SENSOR_FEATURE_SIZE: # Single sample passed
                 sensor_batch_np = sensor_batch_np.reshape(1, -1)
            elif sensor_batch_np.ndim != 2 or sensor_batch_np.shape[1] != config.SENSOR_FEATURE_SIZE:
                 logger.error(f"Invalid sensor_batch shape. Expected (batch, {config.SENSOR_FEATURE_SIZE}), got {sensor_batch_np.shape}")
                 return None
        except Exception as e_conv:
            logger.error(f"Could not convert raw_sensor_batch to NumPy array: {e_conv}")
            return None

        scaled_sensor_batch_np = self._preprocess_input_batch(sensor_batch_np)
        if scaled_sensor_batch_np is None:
            logger.error("Sensor data preprocessing (scaling) failed.")
            return None


        # --- 2. Prepare other modal inputs (Sim Features, Images) ---
        # Sim features might also need scaling if they are numerical and scaler was fit on them.
        # For now, assume they are pre-scaled or don't need scaling.
        sim_batch_np = np.array(sim_feature_batch, dtype=np.float32) if sim_feature_batch is not None else None
        if sim_batch_np is not None and (sim_batch_np.ndim != 2 or sim_batch_np.shape[1] != getattr(config, 'SIMULATION_FEATURE_SIZE', 0)):
            logger.error(f"Invalid sim_feature_batch shape. Expected (batch, {getattr(config, 'SIMULATION_FEATURE_SIZE', 0)}), got {sim_batch_np.shape if sim_batch_np is not None else 'None'}")
            return None

        # Image batch should already be preprocessed (e.g., ToTensor, Normalize) by the data source
        # and ready to be converted to a torch tensor.
        image_batch_tensor = torch.tensor(image_batch, dtype=torch.float32).to(self.device) if image_batch is not None and config.MODEL_USES_IMAGES else None
        if image_batch_tensor is not None and image_batch_tensor.shape[1:] != config.IMAGE_INPUT_SHAPE: # Check C,H,W
             logger.error(f"Invalid image_batch shape. Expected (batch, {config.IMAGE_INPUT_SHAPE}), got {image_batch_tensor.shape}")
             return None


        # --- 3. Perform Inference (PyTorch or ONNX) ---
        model_outputs_np = None
        try:
            if self.using_onnx and self.onnx_session:
                # ONNX Runtime inference
                # Input names must match the ONNX model's expected input names
                # Example: ort_inputs = {'sensor_input': scaled_sensor_batch_np}
                #          if sim_batch_np is not None: ort_inputs['sim_input'] = sim_batch_np
                #          if image_batch_tensor is not None: ort_inputs['image_input'] = image_batch_tensor.cpu().numpy() # ONNX expects numpy
                # For this example, assume the ONNX model has one input named "input"
                # that takes concatenated features or handles them internally based on training.
                # This part is highly dependent on how the ONNX model was exported.
                # Let's assume the ONNX model takes the same inputs as the PyTorch model,
                # and we need to find their names.
                # For simplicity, we'll assume the ONNX model expects a single input if only sensor data is used.
                # A multi-input ONNX model requires named inputs.

                # --- ONNX Input Preparation (Needs to match ONNX model structure) ---
                # This is a placeholder and needs to be adapted to the specific ONNX model.
                # If the PyTorch model's forward() takes separate args, the ONNX model likely expects named inputs.
                ort_inputs = {}
                input_names = [inp.name for inp in self.onnx_session.get_inputs()]

                if 'sensor_data' in input_names and scaled_sensor_batch_np is not None:
                     ort_inputs['sensor_data'] = scaled_sensor_batch_np
                # Add sim_feature_data and image_data if model was exported with these named inputs
                if 'sim_feature_data' in input_names and sim_batch_np is not None:
                     ort_inputs['sim_feature_data'] = sim_batch_np
                if 'image_data' in input_names and image_batch_tensor is not None:
                     ort_inputs['image_data'] = image_batch_tensor.cpu().numpy() # ONNX RT prefers numpy

                # If only one input is expected by ONNX model after concatenation in PyTorch graph:
                # This path is complex if the PyTorch model's `forward` signature is used for export
                # without explicitly naming inputs/outputs for ONNX.
                # A common practice is to have the PyTorch model's forward take a single dict or tuple
                # for easier ONNX export, or name inputs during `torch.onnx.export`.

                # Fallback: If names are generic like 'input_0', 'input_1'
                # This is highly dependent on ONNX export. For this example, we'll assume simple named inputs
                # as defined in the PyTorch model's forward method signature if they match.
                # If it fails, it indicates an issue with ONNX model input names.

                if not ort_inputs: # If no specific named inputs matched and model expects some
                    logger.error("ONNX model input preparation failed: Could not map features to ONNX input names. Check ONNX model's input signature.")
                    return None

                logger.debug(f"ONNX Inference with inputs: {list(ort_inputs.keys())}")
                model_outputs_list = self.onnx_session.run(None, ort_inputs) # Returns a list of output arrays
                model_outputs_np = model_outputs_list[0] # Assuming the first output is the main prediction

            elif self.pytorch_model:
                # PyTorch inference
                sensor_tensor = torch.tensor(scaled_sensor_batch_np, dtype=torch.float32).to(self.device)
                sim_tensor = torch.tensor(sim_batch_np, dtype=torch.float32).to(self.device) if sim_batch_np is not None else None
                # image_batch_tensor is already a tensor on device (or None)

                with torch.no_grad():
                    outputs = self.pytorch_model(
                        sensor_data=sensor_tensor,
                        sim_feature_data=sim_tensor,
                        image_data=image_batch_tensor
                    )
                model_outputs_np = outputs.cpu().numpy()
            else: # Should not happen if model_loaded is True
                logger.error("No valid model (PyTorch or ONNX) session available for inference.")
                return None

        except Exception as e:
            logger.error(f"Error during AI model inference: {e}", exc_info=True)
            return None

        # --- 4. Postprocess Output ---
        batch_predictions = self._postprocess_output_batch(model_outputs_np)
        # logger.debug(f"Batch prediction results: {batch_predictions}")

        # --- 5. Log predictions (Optional, or done by a subscriber to prediction MQ) ---
        if self.data_logger and getattr(config, 'LOG_AI_PREDICTIONS_IN_ENGINE', False):
            for i, pred_result in enumerate(batch_predictions):
                # Create input_data dict for logging based on what was used
                log_input = {"sensor_features_scaled": scaled_sensor_batch_np[i].tolist()}
                if sim_batch_np is not None: log_input["sim_features"] = sim_batch_np[i].tolist()
                # Image data logging is complex, maybe log path or a hash
                self.data_logger.log_ai_prediction(pred_result, input_data=log_input) # Needs robot_id, job_id if available

        return batch_predictions


    # --- Message Queue Integration ---
    def _connect_mq_subscriber(self):
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika: return False
        with self._mq_lock:
            if self._mq_channel_subscriber and self._mq_channel_subscriber.is_open: return True
            try:
                logger.info(f"AI Engine: Connecting MQ Subscriber to {config.MQ_HOST}:{config.MQ_PORT}")
                credentials = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                parameters = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, credentials, heartbeat=600)
                self._mq_connection_subscriber = pika.BlockingConnection(parameters)
                self._mq_channel_subscriber = self._mq_connection_subscriber.channel()
                # Assume sensor data is published to MQ_SENSOR_DATA_EXCHANGE (topic exchange)
                self._mq_channel_subscriber.exchange_declare(exchange=config.MQ_SENSOR_DATA_EXCHANGE, exchange_type='topic', durable=True)
                # Declare an exclusive queue for this engine to consume sensor data
                result = self._mq_channel_subscriber.queue_declare(queue='', exclusive=True)
                queue_name = result.method.queue
                # Bind to relevant sensor data topics (e.g., "sensor.raw.#" or specific processed topics)
                binding_key = f"{config.MQ_SENSOR_DATA_ROUTING_KEY_PREFIX}#" # Consume all raw sensor data
                self._mq_channel_subscriber.queue_bind(exchange=config.MQ_SENSOR_DATA_EXCHANGE, queue=queue_name, routing_key=binding_key)
                logger.info(f"AI Engine: MQ Subscriber connected, queue '{queue_name}' bound to '{binding_key}'.")
                self._mq_channel_subscriber.basic_consume(queue=queue_name, on_message_callback=self._mq_sensor_data_callback, auto_ack=True)
                return True
            except Exception as e:
                logger.error(f"AI Engine: Failed to connect MQ Subscriber: {e}", exc_info=True)
                if self._mq_connection_subscriber and self._mq_connection_subscriber.is_open: self._mq_connection_subscriber.close()
                self._mq_connection_subscriber, self._mq_channel_subscriber = None, None
                return False

    def _connect_mq_publisher(self):
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika: return False
        with self._mq_lock:
            if self._mq_channel_publisher and self._mq_channel_publisher.is_open: return True
            try:
                logger.info(f"AI Engine: Connecting MQ Publisher to {config.MQ_HOST}:{config.MQ_PORT}")
                credentials = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                parameters = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, credentials, heartbeat=600)
                self._mq_connection_publisher = pika.BlockingConnection(parameters)
                self._mq_channel_publisher = self._mq_connection_publisher.channel()
                # Declare exchange for AI predictions (e.g., a topic exchange)
                self._mq_channel_publisher.exchange_declare(exchange=config.MQ_AI_PREDICTION_EXCHANGE, exchange_type='topic', durable=True)
                logger.info(f"AI Engine: MQ Publisher connected and exchange '{config.MQ_AI_PREDICTION_EXCHANGE}' declared.")
                return True
            except Exception as e:
                logger.error(f"AI Engine: Failed to connect MQ Publisher: {e}", exc_info=True)
                if self._mq_connection_publisher and self._mq_connection_publisher.is_open: self._mq_connection_publisher.close()
                self._mq_connection_publisher, self._mq_channel_publisher = None, None
                return False

    def _mq_sensor_data_callback(self, ch, method, properties, body):
        """Callback for processing sensor data messages received from MQ."""
        try:
            # logger.debug(f"MQ Callback: Received message from {method.routing_key}")
            sensor_data_packet = json.loads(body.decode('utf-8'))
            # Put the raw sensor data packet onto the internal processing queue
            # The packet should contain 'robot_id', 'sensor_name', 'value', 'timestamp_utc' etc.
            # It might need further transformation if it's not just a list/array of features.
            # This example assumes sensor_data_packet contains {'robot_id': ..., 'features': [...], ...}
            # For simplicity, assume 'value' is the feature array or relevant data.
            # Needs to match what SensorDataHandler publishes.

            # This example is simplified: it assumes the MQ message body *is* the raw_sensor_batch
            # or can be directly transformed into it. A real system might need more complex parsing
            # and aggregation if sensor data comes in individual readings.
            # For batch inference, we might collect a few messages before processing.

            # For now, put the whole packet on the queue. Worker thread will decide what to use.
            self._incoming_sensor_data_queue.put(sensor_data_packet, timeout=0.5)

        except json.JSONDecodeError:
            logger.error(f"MQ Callback: Failed to decode JSON from sensor data message: {body.decode('utf-8')[:200]}")
        except queue.Full:
            logger.warning("AI Inference input queue is full. Sensor data from MQ might be dropped.")
        except Exception as e:
            logger.error(f"MQ Callback: Error processing sensor data message: {e}", exc_info=True)


    def _inference_worker_loop(self):
        """Worker thread to process sensor data from internal queue and publish predictions."""
        logger.info("AI Inference worker thread started. Waiting for sensor data...")
        while not self._stop_mq_event.is_set():
            try:
                # Get data from internal queue (populated by MQ callback)
                # This allows batching if multiple sensor readings arrive quickly.
                # For true batching, collect multiple items before calling process_sensor_data_batch.
                # This example processes one "packet" at a time, assuming a packet might be a batch.
                sensor_data_packet = self._incoming_sensor_data_queue.get(timeout=1.0) # Wait 1 sec

                # --- Extract features for prediction ---
                # This needs to be robust based on what SensorDataHandler publishes.
                # Example: if sensor_data_packet['value'] is the numpy array of features for one sample
                # For batch, it would be sensor_data_packet['values'] (list of lists or 2D array)
                # For simplicity, assume `process_sensor_data_batch` handles single or batch if passed a list.
                # And sensor_data_packet contains everything needed.
                # Example: features_for_ai = sensor_data_packet.get('features_for_ai_model')
                #          sim_features = sensor_data_packet.get('simulation_features') (if available)
                #          image_data = sensor_data_packet.get('image_data_path_or_tensor') (if available)
                
                # This is a placeholder for feature extraction logic from the sensor_data_packet
                # Assuming 'value' contains the primary sensor data needed for `raw_sensor_batch`
                # And other fields like 'robot_id', 'timestamp_utc' are for context.
                raw_sensor_values = sensor_data_packet.get('value')
                # For multi-modal, you'd extract other features too.
                # raw_sim_values = sensor_data_packet.get('simulation_value')
                # raw_image_ref = sensor_data_packet.get('image_reference')


                if raw_sensor_values is None:
                    logger.warning("Received sensor data packet from queue with no 'value' field. Skipping.")
                    self._incoming_sensor_data_queue.task_done()
                    continue

                # Perform inference (this method handles single or batch if input is shaped correctly)
                # The process_sensor_data_batch expects a batch (even if batch_size=1)
                # Ensure raw_sensor_values is a list of lists or 2D numpy array
                if not isinstance(raw_sensor_values, (list, np.ndarray)): # Basic check
                     logger.warning(f"Skipping inference for packet, 'value' is not list/array: {type(raw_sensor_values)}")
                     self._incoming_sensor_data_queue.task_done()
                     continue

                current_batch_to_process = [raw_sensor_values] if not isinstance(raw_sensor_values[0], list) else raw_sensor_values

                batch_predictions = self.process_sensor_data_batch(current_batch_to_process) # Add sim_feature_batch, image_batch if available

                if batch_predictions:
                    # --- Publish prediction results to MQ ---
                    for i, prediction_result in enumerate(batch_predictions):
                        # Add context from original sensor data packet to the prediction
                        prediction_output_packet = {
                            "timestamp_utc_prediction": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "timestamp_utc_sensor": sensor_data_packet.get('timestamp_utc'),
                            "robot_id": sensor_data_packet.get('robot_id'),
                            "sensor_id": sensor_data_packet.get('sensor_id'),
                            "prediction": prediction_result,
                            "source_system": "AIInferenceEngine"
                        }
                        # Example routing key for predictions: "ai.prediction.robot1.quality_class"
                        pred_routing_key = f"ai.prediction.{'robot'+str(sensor_data_packet.get('robot_id','unknown')) if sensor_data_packet.get('robot_id') else 'global'}"
                        if config.AI_MODEL_TYPE == 'classification':
                             pred_routing_key += f".class_{prediction_result.get('predicted_class')}"
                        else: # Regression
                             pred_routing_key += ".score"

                        if not self._publish_prediction_to_mq(pred_routing_key, prediction_output_packet):
                             logger.error("Failed to publish AI prediction to MQ. It might be lost.")
                             # Implement dead-lettering or other retry for critical predictions.
                else:
                    logger.warning(f"Inference returned no predictions for sensor data from {sensor_data_packet.get('sensor_id')}")


                self._incoming_sensor_data_queue.task_done()

            except queue.Empty:
                pass # Timeout, check stop_event and continue
            except Exception as e:
                logger.error(f"AI Inference worker thread error: {e}", exc_info=True)
                time.sleep(0.5) # Sleep briefly after an error

        logger.info("AI Inference worker thread stopped.")

    def _publish_prediction_to_mq(self, routing_key, prediction_packet):
        """Publishes a single prediction packet to the AI prediction exchange."""
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika: return False
        if not self._mq_channel_publisher or not self._mq_channel_publisher.is_open:
            logger.warning("AI Prediction MQ channel not open. Attempting reconnect for publisher.")
            if not self._connect_mq_publisher():
                logger.error(f"Failed to publish AI prediction, MQ publisher connection error. Message for {routing_key} might be lost.")
                return False
        try:
            message_body_str = json.dumps(prediction_packet, ensure_ascii=False)
            with self._mq_lock: # Protect channel operations
                self._mq_channel_publisher.basic_publish(
                    exchange=config.MQ_AI_PREDICTION_EXCHANGE,
                    routing_key=routing_key,
                    body=message_body_str,
                    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE, content_type='application/json')
                )
            # logger.debug(f"Published AI Prediction: Key='{routing_key}'") # Verbose
            return True
        except Exception as e:
            logger.error(f"Failed to publish AI prediction to MQ. RoutingKey: {routing_key}. Error: {e}", exc_info=True)
            # Consider disconnect/reconnect logic for publisher if errors persist
            if isinstance(e, (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker)):
                 self._disconnect_mq_publisher() # Disconnect publisher on connection error
            return False

    def _disconnect_mq_subscriber(self):
         with self._mq_lock:
             if self._mq_channel_subscriber and self._mq_channel_subscriber.is_open:
                 try: self._mq_channel_subscriber.close() # Close channel first
                 except Exception as e_ch: logger.error(f"AI Engine: Error closing MQ subscriber channel: {e_ch}")
             if self._mq_connection_subscriber and self._mq_connection_subscriber.is_open:
                 try: self._mq_connection_subscriber.close()
                 except Exception as e_conn: logger.error(f"AI Engine: Error closing MQ subscriber connection: {e_conn}")
             self._mq_channel_subscriber, self._mq_connection_subscriber = None, None
             logger.info("AI Engine: MQ Subscriber connection closed.")

    def _disconnect_mq_publisher(self):
         with self._mq_lock:
             if self._mq_channel_publisher and self._mq_channel_publisher.is_open:
                 try: self._mq_channel_publisher.close()
                 except Exception as e_ch: logger.error(f"AI Engine: Error closing MQ publisher channel: {e_ch}")
             if self._mq_connection_publisher and self._mq_connection_publisher.is_open:
                 try: self._mq_connection_publisher.close()
                 except Exception as e_conn: logger.error(f"AI Engine: Error closing MQ publisher connection: {e_conn}")
             self._mq_channel_publisher, self._mq_connection_publisher = None, None
             logger.info("AI Engine: MQ Publisher connection closed.")


    def start_consuming_sensor_data(self):
        """Starts the MQ subscriber thread to consume sensor data and the inference worker."""
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika:
            logger.warning("MQ not enabled or Pika not available. AI Engine will not consume sensor data from MQ.")
            return

        if self._mq_subscriber_thread and self._mq_subscriber_thread.is_alive():
            logger.info("AI Engine MQ sensor data consumer already running.")
            return

        if not (self._mq_channel_subscriber and self._mq_channel_subscriber.is_open):
            if not self._connect_mq_subscriber():
                logger.error("Cannot start sensor data consumer: Failed to connect to MQ subscriber.")
                return

        logger.info("Starting AI Engine MQ sensor data consumer thread...")
        self._stop_mq_event.clear()
        # The actual consumption (channel.start_consuming()) is blocking, so it needs its own thread.
        self._mq_subscriber_thread = threading.Thread(target=self._mq_subscriber_main_loop, name="AIEMQSubscriber", daemon=True)
        self._mq_subscriber_thread.start()

        # Start the inference worker thread that processes data from the internal queue
        if self._inference_worker_thread is None or not self._inference_worker_thread.is_alive():
             self._inference_worker_thread = threading.Thread(target=self._inference_worker_loop, name="AIEInferenceWorker", daemon=True)
             self._inference_worker_thread.start()


    def _mq_subscriber_main_loop(self):
        logger.info("AI Engine MQ subscriber main loop started.")
        while not self._stop_mq_event.is_set():
            try:
                # Ensure connection and channel are good before starting/restarting consuming
                if not (self._mq_channel_subscriber and self._mq_channel_subscriber.is_open):
                    logger.warning("MQ subscriber channel not open in main loop. Attempting to reconnect...")
                    if not self._connect_mq_subscriber():
                        logger.error("MQ subscriber re-connection failed. Waiting before retry.")
                        time.sleep(5) # Wait before retrying connection
                        continue # Retry connection

                # Start consuming messages. This is a blocking call.
                # It will run until channel is closed, connection drops, or stop_consuming is called.
                logger.info("AI Engine: Starting to consume sensor data from MQ...")
                self._mq_channel_subscriber.start_consuming() # Blocking call
                # If start_consuming returns, it means consumption stopped (e.g., by stop_consuming or error)
                logger.info("AI Engine: MQ consumption stopped.")
                # Check if it was due to stop_event or an error that closed the channel
                if self._stop_mq_event.is_set():
                    break # Exit loop if stop was signaled
                # If not due to stop_event, channel might have closed. Loop will try to reconnect.

            except pika.exceptions.ConnectionClosedByBroker:
                 logger.warning("MQ Connection closed by broker. Attempting to reconnect...")
                 self._disconnect_mq_subscriber() # Ensure clean state before reconnect
                 time.sleep(5)
            except pika.exceptions.AMQPChannelError as e_ch_err:
                 logger.error(f"MQ Channel error: {e_ch_err}. Attempting to re-establish channel...")
                 self._disconnect_mq_subscriber()
                 time.sleep(5)
            except pika.exceptions.AMQPConnectionError as e_conn_err:
                 logger.error(f"MQ Connection error: {e_conn_err}. Attempting to reconnect...")
                 self._disconnect_mq_subscriber()
                 time.sleep(5)
            except Exception as e:
                logger.error(f"AI Engine MQ subscriber main loop unexpected error: {e}", exc_info=True)
                self._disconnect_mq_subscriber() # Try to reset connection on unknown error
                time.sleep(5) # Wait before retrying

        logger.info("AI Engine MQ subscriber main loop stopped.")


    def stop_consuming_sensor_data(self):
        """Stops the MQ subscriber thread and inference worker."""
        logger.info("AI Engine: Stopping sensor data consumption and inference worker...")
        self._stop_mq_event.set() # Signal all loops to stop

        # Stop MQ subscriber thread
        if self._mq_subscriber_thread and self._mq_subscriber_thread.is_alive():
            if self._mq_channel_subscriber and self._mq_channel_subscriber.is_open:
                 try:
                     # This needs to be called from a different thread than the one start_consuming is running in
                     # For BlockingConnection, we might need to close connection to stop it from another thread.
                     # Or, schedule_once on IOLoop if using different connection adapter.
                     # Simplest for BlockingConnection: close connection.
                     # self._mq_channel_subscriber.stop_consuming() # May not work as expected with BlockingConnection
                     if self._mq_connection_subscriber and self._mq_connection_subscriber.is_open:
                         # Closing connection will make start_consuming exit.
                         self._mq_connection_subscriber.close() # This is often done from another thread
                 except Exception as e_stop_consume:
                     logger.error(f"Error trying to stop MQ consumption: {e_stop_consume}")

            logger.info("Waiting for AI Engine MQ subscriber thread to join...")
            self._mq_subscriber_thread.join(timeout=5.0)
            if self._mq_subscriber_thread.is_alive():
                logger.warning("AI Engine MQ subscriber thread did not join gracefully.")
        self._mq_subscriber_thread = None

        # Stop inference worker thread (it checks _stop_mq_event)
        if self._inference_worker_thread and self._inference_worker_thread.is_alive():
            logger.info("Waiting for AI Inference worker thread to join...")
            self._inference_worker_thread.join(timeout=5.0)
            if self._inference_worker_thread.is_alive():
                 logger.warning("AI Inference worker thread did not join gracefully.")
        self._inference_worker_thread = None


        self._disconnect_mq_subscriber() # Ensure final cleanup
        self._disconnect_mq_publisher()  # Ensure publisher is also closed
        logger.info("AI Engine: Sensor data consumption and inference worker stopped.")


    def get_status(self):
        """Returns the current status of the AI Inference Engine."""
        return {
            "model_loaded": self.model_loaded,
            "scaler_loaded": self.scaler_loaded,
            "using_onnx": self.using_onnx if self.model_loaded else False,
            "device": str(self.device),
            "mq_subscriber_connected": self._mq_channel_subscriber.is_open if self._mq_channel_subscriber else False,
            "mq_publisher_connected": self._mq_channel_publisher.is_open if self._mq_channel_publisher else False,
            "mq_subscriber_thread_alive": self._mq_subscriber_thread.is_alive() if self._mq_subscriber_thread else False,
            "inference_worker_thread_alive": self._inference_worker_thread.is_alive() if self._inference_worker_thread else False,
            "incoming_data_queue_size": self._incoming_sensor_data_queue.qsize()
        }


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    logger.info("--- AI Inference Engine (Advanced with MQ) Example ---")

    # Dummy Config for MQ, Model Paths, etc.
    class DummyConfigAdvancedAIE:
        USE_MESSAGE_QUEUE = True # Set to False to test without MQ
        MQ_TYPE = 'rabbitmq'
        MQ_HOST = 'localhost'; MQ_PORT = 5672; MQ_USER = 'guest'; MQ_PASSWORD = 'guest'; MQ_VHOST = '/'
        MQ_SENSOR_DATA_EXCHANGE = 'sensor_data_exchange_test_aie'
        MQ_SENSOR_DATA_ROUTING_KEY_PREFIX = 'sensor.test.aie.'
        MQ_AI_PREDICTION_EXCHANGE = 'ai_predictions_exchange_test_aie'

        # Assume create_dummy_model_files.py was run to create these
        MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models') # Adjust path if needed
        MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'welding_ai_model_v2.pth')
        SCALER_SAVE_PATH = os.path.join(MODELS_DIR, 'sensor_data_scaler_v2.pkl')
        ONNX_MODEL_PATH = os.path.join(MODELS_DIR, 'welding_ai_model_v2.onnx')
        USE_ONNX_FOR_INFERENCE = False # Set to True to test ONNX if model exists

        SENSOR_FEATURE_SIZE = 4
        SIMULATION_FEATURE_SIZE = 0 # No sim features in this basic test
        MODEL_USES_IMAGES = False
        IMAGE_INPUT_SHAPE = (1,1,1)
        OUTPUT_CLASSES = 4
        AI_MODEL_TYPE = 'classification'
        LOG_LEVEL = 'DEBUG'
        # DATABASE_PATH = 'test_aie_log.db' # For dummy data logger

    config = DummyConfigAdvancedAIE()

    # Ensure dummy models/scaler exist for testing (run create_dummy_model_files.py first)
    if not os.path.exists(config.MODEL_SAVE_PATH) or not os.path.exists(config.SCALER_SAVE_PATH):
        logger.error(f"Dummy model ({config.MODEL_SAVE_PATH}) or scaler ({config.SCALER_SAVE_PATH}) not found. Please create them first.")
        # sys.exit(1) # Optionally exit if files are crucial for the test
        logger.warning("Proceeding without model/scaler, AI engine might not load them.")


    # Dummy DataLoggerDB for logging predictions if enabled
    # dummy_data_logger = DataLoggerDB() # Get singleton

    # --- Initialize AI Inference Engine ---
    ai_engine = AIInferenceEngine(data_logger=None) # Pass data_logger if using it for predictions

    # --- Test direct batch processing (if MQ is disabled or for specific tests) ---
    if not config.USE_MESSAGE_QUEUE or not ai_engine.model_loaded:
        logger.info("\n--- Testing Direct Batch Inference (MQ might be disabled or model not loaded) ---")
        if ai_engine.model_loaded:
            dummy_batch_sensor_data = np.random.rand(5, config.SENSOR_FEATURE_SIZE).astype(np.float32) # Batch of 5 samples
            predictions = ai_engine.process_sensor_data_batch(dummy_batch_sensor_data)
            if predictions:
                logger.info(f"Direct batch predictions ({len(predictions)}):")
                for i, p in enumerate(predictions):
                    logger.info(f" Sample {i}: {p}")
            else:
                logger.error("Direct batch inference failed.")
        else:
            logger.warning("Skipping direct batch inference test as model is not loaded.")


    # --- Test MQ consumption and prediction publishing ---
    if config.USE_MESSAGE_QUEUE and pika and ai_engine.model_loaded:
        logger.info("\n--- Testing MQ Sensor Data Consumption and Prediction Publishing ---")

        # Start consuming sensor data from MQ
        ai_engine.start_consuming_sensor_data()

        # MQ Publisher (to simulate SensorDataHandler publishing sensor data)
        mq_publisher_conn = None
        mq_publisher_channel = None
        try:
            logger.info("[TestPublisher] Connecting to RabbitMQ to send dummy sensor data...")
            credentials = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
            parameters = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, credentials)
            mq_publisher_conn = pika.BlockingConnection(parameters)
            mq_publisher_channel = mq_publisher_conn.channel()
            mq_publisher_channel.exchange_declare(exchange=config.MQ_SENSOR_DATA_EXCHANGE, exchange_type='topic', durable=True)
            logger.info("[TestPublisher] Connected. Will send a few sensor data packets.")

            for i in range(3): # Send 3 dummy sensor packets
                dummy_sensor_packet = {
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "robot_id": 1,
                    "sensor_id": f"test_sensor_mq_{i}",
                    "sensor_name": "dummy_mq_sensor",
                    "value_type": "real_array", # Assuming 'value' is a list/array of features
                    "value": np.random.rand(config.SENSOR_FEATURE_SIZE).astype(np.float32).tolist(), # List of floats
                    "unit": "N/A"
                }
                routing_key = f"{config.MQ_SENSOR_DATA_ROUTING_KEY_PREFIX}robot1.dummy_mq_sensor"
                mq_publisher_channel.basic_publish(
                    exchange=config.MQ_SENSOR_DATA_EXCHANGE,
                    routing_key=routing_key,
                    body=json.dumps(dummy_sensor_packet),
                    properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
                )
                logger.info(f"[TestPublisher] Sent dummy sensor data packet {i+1} with key '{routing_key}'.")
                time.sleep(0.2) # Brief pause

        except Exception as e_pub_test:
            logger.error(f"[TestPublisher] Error: {e_pub_test}")
        finally:
            if mq_publisher_channel and mq_publisher_channel.is_open: mq_publisher_channel.close()
            if mq_publisher_conn and mq_publisher_conn.is_open: mq_publisher_conn.close()
            logger.info("[TestPublisher] Connection closed.")


        # Let AI engine process for a few seconds
        logger.info("Waiting for AI engine to process MQ messages and publish predictions (approx 5s)...")
        time.sleep(5) # Give time for MQ messages to be consumed, processed, and predictions published

    else:
        logger.warning("Skipping MQ test: MQ disabled, Pika not available, or AI model not loaded.")


    # --- Get Final Status and Shutdown ---
    logger.info(f"\nFinal AI Engine Status: {ai_engine.get_status()}")

    logger.info("Shutting down AI Inference Engine...")
    ai_engine.stop_consuming_sensor_data() # Stops MQ threads and worker

    # if dummy_data_logger: dummy_data_logger.close_connection()

    logger.info("--- AI Inference Engine (Advanced with MQ) Example Finished ---")