# src/config.py
# Description: (REVISED & ADVANCED) Configuration settings for the Samsung E&A Digital Twin Welding AI System.
#              Loads configurations from environment variables and external JSON files.

import os
import json
import logging # Import logging here to configure it once centrally

# --- Project Root and Configuration File Directory ---
# Assuming this config.py is in src/, project_root is one level up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_FILES_DIR = os.path.join(os.path.dirname(__file__), 'config_files') # Directory for JSON configs

# --- Basic Logging Configuration (Configure once here) ---
LOG_LEVEL_STR = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, 'logs', 'digital_twin_system.log')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(threadName)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__) # Root logger for this config, modules should get their own
logger.info(f"Logging configured. Level: {LOG_LEVEL_STR}, File: {LOG_FILE_PATH}")


# --- Helper function to load JSON configs ---
def load_json_config(file_name, default_value=None):
    file_path = os.path.join(CONFIG_FILES_DIR, file_name)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.info(f"Successfully loaded JSON config from: {file_path}")
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load JSON config from {file_path}: {e}", exc_info=True)
            return default_value if default_value is not None else {} # Return default on error
    else:
        logger.warning(f"JSON config file not found: {file_path}. Using default: {default_value}")
        return default_value if default_value is not None else {}


# --- Load External JSON Configurations ---
ROBOT_CONFIGS = load_json_config('robot_configs.json', default_value=[])
ADAPTIVE_CONTROL_RULES = load_json_config('adaptive_rules.json', default_value={})
WELDING_RECIPES = load_json_config('welding_recipes.json', default_value={}) # Used by WeldingProcessManager


# --- Input Parameters (Could also be part of a recipe or material database) ---
PARAM_RANGES = { # Default ranges if not specified elsewhere
    'current': (float(os.getenv('PARAM_CURRENT_MIN', 80)), float(os.getenv('PARAM_CURRENT_MAX', 200))),
    'voltage': (float(os.getenv('PARAM_VOLTAGE_MIN', 15)), float(os.getenv('PARAM_VOLTAGE_MAX', 30))),
    'speed': (float(os.getenv('PARAM_SPEED_MIN', 100)), float(os.getenv('PARAM_SPEED_MAX', 500))),
    'gas_flow': (float(os.getenv('PARAM_GAS_MIN', 15)), float(os.getenv('PARAM_GAS_MAX', 25))),
    'heat_input': (None, None) # Calculated, or could have target ranges
}
ADJUSTMENT_PARAMS = { # For parameters like torch angle, CTWD
    'torch_angle': (float(os.getenv('PARAM_TORCH_ANGLE_MIN', 0)), float(os.getenv('PARAM_TORCH_ANGLE_MAX', 45))),
    'ctwd': (float(os.getenv('PARAM_CTWD_MIN', 10)), float(os.getenv('PARAM_CTWD_MAX', 20)))
}

# --- Data Paths & Management ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Historical/Batch Data (for AI training)
SENSOR_DATA_CSV_PATH = os.path.join(DATA_DIR, os.getenv('SENSOR_CSV_FILENAME', 'historical_sensor_data.csv'))
LABEL_DATA_CSV_PATH = os.path.join(DATA_DIR, os.getenv('LABEL_CSV_FILENAME', 'historical_labels.csv'))
DUMMY_START_TIME = os.getenv('DUMMY_DATA_START_TIME', '2024-01-01T00:00:00+00:00') # For dummy data scripts

# Unreal Engine Simulated Data Path
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, 'simulated_datasets')
os.makedirs(SIMULATED_DATA_DIR, exist_ok=True)

# Image Data Directory (if used by AI model)
IMAGE_DATA_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGE_DATA_DIR, exist_ok=True)

# --- Database Configuration ---
DB_TYPE = os.getenv('DB_TYPE', 'sqlite') # 'postgresql', 'sqlite', 'mongodb', 'influxdb'
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'welding_dt_db')
DB_USER = os.getenv('DB_USER', 'wdt_admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'P@$$wOrdN33d$T0BeStr0ng!') # Use secure way to manage passwords in prod
SQLITE_DB_PATH = os.path.join(DATA_DIR, 'welding_system_log.db') # For SQLite
DB_LOG_BATCH_SIZE = int(os.getenv('DB_LOG_BATCH_SIZE', '100'))
DB_LOG_BATCH_TIMEOUT_SEC = float(os.getenv('DB_LOG_BATCH_TIMEOUT_SEC', '5.0'))

# --- Message Queue Configuration (RabbitMQ Example) ---
USE_MESSAGE_QUEUE = os.getenv('USE_MESSAGE_QUEUE', 'True').lower() == 'true'
MQ_TYPE = os.getenv('MQ_TYPE', 'rabbitmq')
MQ_HOST = os.getenv('MQ_HOST', 'localhost')
MQ_PORT = int(os.getenv('MQ_PORT', '5672'))
MQ_USER = os.getenv('MQ_USER', 'guest')
MQ_PASSWORD = os.getenv('MQ_PASSWORD', 'guest')
MQ_VHOST = os.getenv('MQ_VHOST', '/')
MQ_HEARTBEAT_SEC = int(os.getenv('MQ_HEARTBEAT_SEC', '600'))

# Message Queue Exchanges and Queues/Routing Keys (Examples)
MQ_ROBOT_COMMAND_EXCHANGE = os.getenv('MQ_ROBOT_COMMAND_EXCHANGE', 'robot_commands_ex') # Direct exchange
MQ_ROBOT_COMMAND_QUEUE_PREFIX = os.getenv('MQ_ROBOT_COMMAND_QUEUE_PREFIX', 'robot_cmd_q_robot') # e.g., robot_cmd_q_robot1
MQ_ROBOT_RESPONSE_EXCHANGE = os.getenv('MQ_ROBOT_RESPONSE_EXCHANGE', 'robot_responses_ex') # Direct exchange for replies
MQ_ROBOT_STATUS_EXCHANGE = os.getenv('MQ_ROBOT_STATUS_EXCHANGE', 'robot_status_ex') # Topic exchange
MQ_ROBOT_STATUS_ROUTING_KEY_PREFIX = os.getenv('MQ_ROBOT_STATUS_ROUTING_KEY_PREFIX', 'robot.status.') # e.g., robot.status.robot1

MQ_SENSOR_DATA_EXCHANGE = os.getenv('MQ_SENSOR_DATA_EXCHANGE', 'sensor_data_ex') # Topic exchange
MQ_SENSOR_DATA_ROUTING_KEY_PREFIX = os.getenv('MQ_SENSOR_DATA_ROUTING_KEY_PREFIX', 'sensor.data.') # e.g., sensor.data.robot1.temperature

MQ_AI_REQUEST_EXCHANGE = os.getenv('MQ_AI_REQUEST_EXCHANGE', 'ai_requests_ex') # Direct or Topic
MQ_AI_PREDICTION_EXCHANGE = os.getenv('MQ_AI_PREDICTION_EXCHANGE', 'ai_predictions_ex') # Topic exchange

MQ_QEA_RESULTS_EXCHANGE = os.getenv('MQ_QEA_RESULTS_EXCHANGE', 'qea_results_ex') # Topic exchange (for evaluations and control commands)
# Routing key for QEA publishing control commands: e.g., "qea.control_command.robot1.parameter_adjustment"
# Routing key for QEA publishing evaluations: e.g., "qea.evaluation.robot1.good"

# --- Unreal Engine Simulator/Visualizer Interface ---
ENABLE_PHYSICS_INTERFACE = os.getenv('ENABLE_PHYSICS_INTERFACE', 'True').lower() == 'true'
UE_SIMULATOR_IP = os.getenv('UE_SIMULATOR_IP', '127.0.0.1')
UE_SIMULATOR_PORT = int(os.getenv('UE_SIMULATOR_PORT', '9999'))
UE_RESPONSE_TIMEOUT_SEC = int(os.getenv('UE_RESPONSE_TIMEOUT_SEC', '30'))
UE_RECONNECT_INTERVAL_SEC = int(os.getenv('UE_RECONNECT_INTERVAL_SEC', '5'))
UE_MAX_RECONNECT_ATTEMPTS = int(os.getenv('UE_MAX_RECONNECT_ATTEMPTS', '3'))
POSE_STREAM_INTERVAL_SEC = float(os.getenv('POSE_STREAM_INTERVAL_SEC', str(1 / 30.0))) # Target 30Hz

# --- Robot Control Interface ---
ENABLE_ROBOT_INTERFACE = os.getenv('ENABLE_ROBOT_INTERFACE', 'True').lower() == 'true'
ROBOT_STATUS_MONITOR_INTERVAL_SEC = float(os.getenv('ROBOT_STATUS_MONITOR_INTERVAL_SEC', '0.05'))
ROBOT_STATUS_RECEIVE_TIMEOUT_SEC = float(os.getenv('ROBOT_STATUS_RECEIVE_TIMEOUT_SEC', '0.1'))
ROBOT_MONITOR_SHUTDOWN_TIMEOUT_SEC = float(os.getenv('ROBOT_MONITOR_SHUTDOWN_TIMEOUT_SEC', '2.0'))
ROBOT_COMMAND_TIMEOUT_SEC = int(os.getenv('ROBOT_COMMAND_TIMEOUT_SEC', '10'))
USE_DUMMY_ROBOTS = os.getenv('USE_DUMMY_ROBOTS', 'False').lower() == 'true'


# --- AI Model Configuration ---
ENABLE_AI_INFERENCE = os.getenv('ENABLE_AI_INFERENCE', 'True').lower() == 'true'
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, os.getenv('MODEL_FILENAME', 'welding_ai_model.pth'))
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, os.getenv('SCALER_FILENAME', 'sensor_data_scaler.pkl'))
USE_ONNX_FOR_INFERENCE = os.getenv('USE_ONNX_FOR_INFERENCE', 'False').lower() == 'true'
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, os.getenv('ONNX_MODEL_FILENAME', 'welding_ai_model.onnx'))

# AI Model Architecture Parameters (from previous advanced ai_model.py)
SENSOR_FEATURE_SIZE = int(os.getenv('SENSOR_FEATURE_SIZE', '4'))
SIMULATION_FEATURE_SIZE = int(os.getenv('SIMULATION_FEATURE_SIZE', '0')) # Default to 0 if not using
MODEL_USES_IMAGES = os.getenv('MODEL_USES_IMAGES', 'False').lower() == 'true'
_img_shape_str = os.getenv('IMAGE_INPUT_SHAPE', '1,128,128').split(',')
IMAGE_INPUT_SHAPE = tuple(map(int, _img_shape_str)) if len(_img_shape_str) == 3 else (1, 128, 128)
AI_MODEL_TYPE = os.getenv('AI_MODEL_TYPE', 'classification') # 'classification' or 'regression'
OUTPUT_CLASSES = int(os.getenv('OUTPUT_CLASSES', '4')) if AI_MODEL_TYPE == 'classification' else 1
# Hidden layer dimensions from environment or default lists
MODEL_SENSOR_HIDDEN_DIMS = json.loads(os.getenv('MODEL_SENSOR_HIDDEN_DIMS', '[128, 64]'))
MODEL_SIM_HIDDEN_DIMS = json.loads(os.getenv('MODEL_SIM_HIDDEN_DIMS', '[64, 32]'))
MODEL_IMAGE_CNN_CHANNELS = json.loads(os.getenv('MODEL_IMAGE_CNN_CHANNELS', '[16, 32]'))
MODEL_IMAGE_FC_HIDDEN_DIMS = json.loads(os.getenv('MODEL_IMAGE_FC_HIDDEN_DIMS', '[128, 64]'))
MODEL_COMBINED_FC_HIDDEN_DIMS = json.loads(os.getenv('MODEL_COMBINED_FC_HIDDEN_DIMS', '[128, 64]'))
MODEL_DROPOUT_RATE = float(os.getenv('MODEL_DROPOUT_RATE', '0.3'))

# AI Training Parameters
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0005'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '100'))
TRAINING_LOG_BATCH_INTERVAL = int(os.getenv('TRAINING_LOG_BATCH_INTERVAL', '50'))
TRAINING_WEIGHT_DECAY = float(os.getenv('TRAINING_WEIGHT_DECAY', '0.01'))
TRAINING_USE_EARLY_STOPPING = os.getenv('TRAINING_USE_EARLY_STOPPING', 'True').lower() == 'true'
TRAINING_EARLY_STOPPING_PATIENCE = int(os.getenv('TRAINING_EARLY_STOPPING_PATIENCE', '10'))
TRAINING_EARLY_STOPPING_MIN_DELTA = float(os.getenv('TRAINING_EARLY_STOPPING_MIN_DELTA', '0.0001')) # Smaller delta
TRAINING_LR_SCHEDULER_USE = os.getenv('TRAINING_LR_SCHEDULER_USE', 'True').lower() == 'true'
TRAINING_LR_SCHEDULER_TYPE = os.getenv('TRAINING_LR_SCHEDULER_TYPE', 'ReduceLROnPlateau')
TRAINING_LR_FACTOR = float(os.getenv('TRAINING_LR_FACTOR', '0.1'))
TRAINING_LR_PATIENCE = int(os.getenv('TRAINING_LR_PATIENCE', '5'))

# Distributed Training Config
DISTRIBUTED_BACKEND = os.getenv('DISTRIBUTED_BACKEND', 'nccl' if torch.cuda.is_available() else 'gloo')
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
RANK = int(os.getenv('RANK', '0'))
MASTER_ADDR = os.getenv('MASTER_ADDR', 'localhost')
MASTER_PORT = os.getenv('MASTER_PORT', '29500')

# TensorBoard Logging
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'tensorboard_runs')
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# --- Quality Evaluation & Adaptive Control ---
ENABLE_ADAPTIVE_CONTROL = os.getenv('ENABLE_ADAPTIVE_CONTROL', 'True').lower() == 'true'
ADAPTIVE_CONTROL_CYCLE_TIME_SEC = float(os.getenv('ADAPTIVE_CONTROL_CYCLE_TIME_SEC', '0.5'))
QEA_INPUT_QUEUE_SIZE = int(os.getenv('QEA_INPUT_QUEUE_SIZE', '100'))
# Quality Class mapping (can be overridden by a more specific map if needed)
QUALITY_CLASSES_MAP = {
    0: "Complete Fusion / Good", 1: "Incomplete Fusion / Lack of Fusion",
    2: "Undercut", 3: "Hot Tear / Crack"
}
# Thresholds for evaluation logic (examples)
QEA_MIN_AI_CONFIDENCE_THRESHOLD = float(os.getenv('QEA_MIN_AI_CONFIDENCE_THRESHOLD', '0.5'))
SENSOR_TEMP_THRESHOLD_CRITICAL = float(os.getenv('SENSOR_TEMP_THRESHOLD_CRITICAL', '450')) # Celsius
SENSOR_ARC_STABILITY_THRESHOLD_WARN = float(os.getenv('SENSOR_ARC_STABILITY_THRESHOLD_WARN', '0.6')) # Index
USE_PHYSICS_IN_EVALUATION = os.getenv('USE_PHYSICS_IN_EVALUATION', 'False').lower() == 'true'
PHYSICS_SCORE_THRESHOLD_WARN = float(os.getenv('PHYSICS_SCORE_THRESHOLD_WARN', '0.6'))


# --- System Manager & Process Manager ---
SYSTEM_HEALTH_CHECK_INTERVAL_SEC = int(os.getenv('SYSTEM_HEALTH_CHECK_INTERVAL_SEC', '5'))
MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '1')) # WPM handles one job at a time by default
WELDING_RECIPES_PATH = os.path.join(CONFIG_FILES_DIR, 'welding_recipes.json') # Path for WPM to load recipes

# --- HMI Configuration ---
ENABLE_HMI = os.getenv('ENABLE_HMI', 'True').lower() == 'true' # If SystemManager should manage HMI process
HMI_FLASK_HOST = os.getenv('HMI_FLASK_HOST', '0.0.0.0')
HMI_FLASK_PORT = int(os.getenv('HMI_FLASK_PORT', '5000'))
# HMI_API_ENDPOINT = os.getenv('HMI_API_ENDPOINT', 'http://localhost:8000/api/v1') # If HMI calls a separate API Gateway

# --- Sensor Handler & AI Inference Engine Queues ---
ENABLE_SENSOR_HANDLER = os.getenv('ENABLE_SENSOR_HANDLER', 'True').lower() == 'true'
USE_DUMMY_SENSORS = os.getenv('USE_DUMMY_SENSORS', 'False').lower() == 'true'
AI_INFERENCE_INPUT_QUEUE_SIZE = int(os.getenv('AI_INFERENCE_INPUT_QUEUE_SIZE', '200'))
DATALOADER_NUM_WORKERS = int(os.getenv('DATALOADER_NUM_WORKERS', '0')) # For PyTorch DataLoader
DATALOADER_PIN_MEMORY = os.getenv('DATALOADER_PIN_MEMORY', str(torch.cuda.is_available())).lower() == 'true'

# --- Sanity Check (Optional, for debugging) ---
logger.debug(f"CONFIG: Project Root = {PROJECT_ROOT}")
logger.debug(f"CONFIG: Using Database Type = {DB_TYPE}")
logger.debug(f"CONFIG: Using Message Queue = {USE_MESSAGE_QUEUE} ({MQ_TYPE})")
logger.debug(f"CONFIG: Robot Configs Loaded = {len(ROBOT_CONFIGS) > 0}")
logger.debug(f"CONFIG: Adaptive Control Rules Loaded = {len(ADAPTIVE_CONTROL_RULES) > 0}")
logger.debug(f"CONFIG: Welding Recipes Loaded = {len(WELDING_RECIPES) > 0}")