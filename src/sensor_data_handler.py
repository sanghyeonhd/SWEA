# sensor_data_handler.py
# Description: Handles real-time collection and preprocessing of sensor data from various sources.
#              Provides data to AI inference engine, quality evaluator, etc.

import socket
import json
import time
import threading
import logging
import queue # For potential internal buffering or passing data to consumers
import random # For dummy data generation
import datetime # For timestamps

from src import config # Requires SENSOR_CONFIGS in config.py
# from data_logger_db import DataLoggerDB # Use the singleton instance directly or pass it

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Virtual Sensor Communication Protocol (Placeholder) ---
# Assume sensors either push data via TCP/UDP or are polled.
# For simplicity here, we simulate polling or receiving data periodically in separate threads.
# Data Format (Example): { "timestamp": "ISO8601_UTC", "robot_id": 1, "sensor_name": "temperature", "value": 350.5 }
# Or for complex data: { "timestamp": "...", "robot_id": 1, "sensor_name": "vision_image", "value_type": "path", "value": "/path/to/image.png", "extra_info": {...} }


class SensorCollector:
    """Manages collection from a single sensor source."""
    def __init__(self, sensor_config, data_queue, data_logger=None):
        self.sensor_config = sensor_config
        self.sensor_name = sensor_config.get('name', 'unknown_sensor')
        self.robot_id = sensor_config.get('robot_id', None) # Associate sensor with a robot if applicable
        self.data_queue = data_queue # Queue to put collected data for processing
        self.data_logger = data_logger # DataLoggerDB singleton instance

        self._running = False
        self._collection_thread = None

        logger.info(f"Sensor Collector '{self.sensor_name}' initialized (Robot {self.robot_id}).")

    def start(self):
        """Starts the data collection thread for this sensor."""
        if self._running:
            logger.warning(f"Sensor Collector '{self.sensor_name}' is already running.")
            return
        self._running = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info(f"Sensor Collector '{self.sensor_name}' thread started.")

    def stop(self):
        """Stops the data collection thread."""
        if self._running:
            logger.info(f"Sensor Collector '{self.sensor_name}' stopping...")
            self._running = False
            if self._collection_thread and self._collection_thread.is_alive():
                self._collection_thread.join(timeout=5)
                if self._collection_thread.is_alive():
                    logger.warning(f"Sensor Collector '{self.sensor_name}' thread did not terminate gracefully.")
            logger.info(f"Sensor Collector '{self.sensor_name}' stopped.")

    def _collect_loop(self):
        """The main loop for collecting data from a single sensor."""
        collection_interval = self.sensor_config.get('interval_sec', 0.1) # How often to collect
        protocol = self.sensor_config.get('protocol', 'dummy')
        # Add connection logic here based on protocol (TCP, UDP, Serial, etc.)
        # self._establish_connection() # Hypothetical connection setup

        while self._running:
            start_time = time.time()
            collected_data = None

            try:
                # --- Simulate Data Collection (Placeholder Logic) ---
                if protocol == 'dummy':
                    # Generate dummy data based on sensor name
                    value = None
                    value_type = 'real'
                    extra_info = None
                    if self.sensor_name == 'welder_current':
                         value = random.uniform(100, 200)
                    elif self.sensor_name == 'welder_voltage':
                         value = random.uniform(18, 28)
                    elif self.sensor_name == 'welder_speed': # Assuming wire feed speed or robot speed feedback
                         value = random.uniform(200, 400)
                    elif self.sensor_name == 'temperature':
                         value = random.uniform(250, 450) # Example temperature range
                    elif self.sensor_name == 'arc_stability_index':
                         value = random.uniform(0.5, 0.95) # Example index
                    elif self.sensor_name == 'vision_image':
                         value_type = 'path'
                         value = f"data/images/{self.robot_id}_{self.sensor_name}_{int(time.time())}.png" # Dummy path
                         # In a real scenario, you might capture image and save it here or elsewhere
                    # Add more sensor types...

                    if value is not None:
                         collected_data = {
                             "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "robot_id": self.robot_id,
                             "sensor_name": self.sensor_name,
                             "value_type": value_type, # 'real', 'integer', 'text', 'json', 'path', 'bytes'
                             "value": value,
                             "extra_info": extra_info
                         }
                # --- End Simulate Data Collection ---

                # --- Preprocessing (Basic Placeholder) ---
                # Add sensor-specific preprocessing here (e.g., unit conversion, filtering, validation)
                if collected_data and collected_data['value_type'] == 'real':
                     # Example: Ensure value is within a reasonable range
                     if self.sensor_name == 'temperature' and collected_data['value'] > 1000:
                          logger.warning(f"Sensor {self.sensor_name}: Unrealistic temperature reading {collected_data['value']}. Skipping.")
                          collected_data = None # Discard potentially bad data

                # --- End Preprocessing ---


                # --- Log Data (via DataLoggerDB singleton) ---
                if collected_data:
                    if self.data_logger:
                        try:
                            # Note: log_sensor_reading expects sensor_value to be REAL.
                            # If value_type is not 'real', store it in extra_info or another dedicated column.
                            # Adjust logging call based on DataLoggerDB schema.
                            log_value = collected_data.get('value') if collected_data.get('value_type') == 'real' else None
                            log_extra_info = collected_data if collected_data.get('value_type') != 'real' else collected_data.get('extra_info')

                            self.data_logger.log_sensor_reading(
                                sensor_name=collected_data['sensor_name'],
                                sensor_value=log_value, # Store real values here
                                robot_id=collected_data['robot_id'],
                                # job_id=... # Need mechanism to get current job ID
                                extra_info=log_extra_info # Store other data types here
                            )
                        except Exception as e:
                             logger.error(f"Sensor {self.sensor_name}: Error logging data: {e}")
                    else:
                         # logger.debug(f"Sensor {self.sensor_name}: DataLoggerDB not available, collected data: {collected_data}")
                         pass # Log message is sufficient if logging is critical

                    # --- Put data into a queue for main handler/consumers ---
                    # The main SensorDataHandler class or consumers will process this queue
                    self.data_queue.put(collected_data)

            except Exception as e:
                logger.error(f"Sensor Collector '{self.sensor_name}': Error during collection cycle: {e}")
                # Consider handling specific exceptions (e.g., connection errors)

            # Control loop frequency
            elapsed_time = time.time() - start_time
            sleep_time = collection_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)


class SensorDataHandler:
    """
    Manages multiple SensorCollector instances and aggregates/distributes data.
    """
    _instance = None # Singleton pattern
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
         with cls._lock:
             if cls._instance is None:
                 cls._instance = super().__new__(cls)
                 cls._instance._initialized = False # Use internal flag for init
         return cls._instance

    def __init__(self, data_logger=None):
         # Singleton initialization check
         with self._lock:
             if self._initialized:
                 return
             self._initialized = True

             logger.info("Sensor Data Handler initializing...")

             # Get DataLoggerDB singleton instance if not provided
             self.data_logger = data_logger # Assume DataLoggerDB instance is passed
             # Or get singleton: self.data_logger = DataLoggerDB()

             # Load sensor configurations from config.py
             # Assuming config.py has a list like:
             # SENSOR_CONFIGS = [
             #     {'name': 'welder_current', 'type': 'analog', 'protocol': 'tcp', 'robot_id': 1, 'ip': '...', 'port': ..., 'interval_sec': 0.05},
             #     {'name': 'vision_image', 'type': 'digital', 'protocol': 'udp', 'robot_id': 2, 'ip': '...', 'port': ..., 'interval_sec': 0.5},
             #     {'name': 'temperature', 'type': 'analog', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 1.0},
             # ]
             self.sensor_configs = getattr(config, 'SENSOR_CONFIGS', [])
             if not self.sensor_configs:
                 logger.warning("No sensor configurations found in config.py (looking for SENSOR_CONFIGS). Sensor data collection will be non-functional.")

             # --- Internal Data Storage & Queues ---
             # This can store the latest data from each sensor, or a window of recent data
             self._latest_data = {} # { (robot_id, sensor_name): { 'timestamp': '...', 'value': ..., ... } }
             self._data_queue = queue.Queue() # Queue for raw collected data to be processed by handler thread

             self._collectors = [] # List of SensorCollector instances
             self._processing_thread = None
             self._running = False
             self._stop_event = threading.Event()

             # Create SensorCollector instances for each configured sensor
             for sensor_cfg in self.sensor_configs:
                 collector = SensorCollector(sensor_cfg, self._data_queue, self.data_logger)
                 self._collectors.append(collector)

             logger.info(f"Sensor Data Handler initialized with {len(self._collectors)} collectors.")


    def start_collection(self):
        """Starts all sensor collector threads and the internal processing thread."""
        if self._running:
            logger.warning("Sensor Data Handler is already running collection.")
            return

        logger.info("Starting sensor data collection...")
        self._running = True
        self._stop_event.clear()

        # Start individual sensor collectors
        for collector in self._collectors:
            collector.start()

        # Start the internal data processing thread
        self._processing_thread = threading.Thread(target=self._process_data_loop, daemon=True)
        self._processing_thread.start()

        logger.info("Sensor data collection started.")


    def stop_collection(self):
        """Stops all sensor collector threads and the internal processing thread."""
        if not self._running:
            logger.warning("Sensor Data Handler is not running collection.")
            return

        logger.info("Stopping sensor data collection...")
        self._running = False
        self._stop_event.set() # Signal processing thread to stop

        # Stop individual sensor collectors (waits for their threads to join)
        for collector in self._collectors:
            collector.stop()

        # Wait for the internal processing thread to finish
        if self._processing_thread and self._processing_thread.is_alive():
            logger.info("Waiting for data processing thread to complete...")
            self._processing_thread.join(timeout=5) # Give it time
            if self._processing_thread.is_alive():
                logger.warning("Data processing thread did not terminate gracefully.")

        logger.info("Sensor data collection stopped.")


    def _process_data_loop(self):
        """Internal thread loop to process data from the queue."""
        logger.info("Sensor Data Handler processing thread started.")
        while self._running or not self._data_queue.empty():
            try:
                # Get data from the queue (with a timeout so we can check stop_event)
                collected_data = self._data_queue.get(timeout=0.1)

                # --- Aggregate Latest Data ---
                # This is a simple aggregation storing the latest received value for each sensor.
                # For time series or window-based aggregation, more complex logic is needed here.
                key = (collected_data['robot_id'], collected_data['sensor_name'])
                self._latest_data[key] = collected_data # Store the raw collected data structure
                # logger.debug(f"Stored latest data for {key}: {collected_data['value']}")

                # --- Data Synchronization & Distribution (Placeholder) ---
                # In a real system, based on this new data point, you might:
                # 1. Check if you have a synchronized set of data points from multiple sensors
                #    within a small timestamp window.
                # 2. Aggregate these synchronized data points into a structure ready for AI/Evaluation.
                # 3. Pass this aggregated structure to the AI inference engine and Quality Evaluator
                #    (e.g., by calling their methods, or putting into their input queues).
                # 4. Possibly log the *aggregated* data for analysis.

                # Example: If this was 'welder_current', check if you also recently got 'welder_voltage', etc.
                # If yes, create the AI input vector and pass it.
                # This requires knowledge of which sensors constitute the AI input (from config).
                # For simplicity in get_latest_aggregated_data, we just provide direct access below.

                # logger.debug(f"Processed data for {key}")
                self._data_queue.task_done() # Mark task done for queue management

            except queue.Empty:
                # Queue was empty within the timeout, continue loop and check stop_event
                pass
            except Exception as e:
                logger.error(f"Sensor Data Handler processing error: {e}", exc_info=True)

            # Check stop event periodically even if queue is not empty
            if self._stop_event.is_set():
                break

        logger.info("Sensor Data Handler processing thread stopped.")


    def get_latest_aggregated_data(self, robot_id=None):
        """
        Retrieves the latest received data from all relevant sensors, optionally filtered by robot_id.
        Aggregates sensor data into structures needed by consumers (AI, Evaluator).

        Args:
            robot_id (int, optional): Filter data for a specific robot. If None, get data from all relevant robots.

        Returns:
            dict: Aggregated data structures. Example:
                  {'robot_id': 1,
                   'timestamp': '...', # Latest timestamp among aggregated data
                   'values_for_ai': [current, voltage, speed, temperature], # numpy array or list matching AI input size
                   'values_for_evaluator': {'temperature': 350, 'arc_stability_index': 0.8}, # More detailed sensor data
                  }
                  Returns None if essential data is missing or handler is not running.
        """
        if not self._running:
             logger.warning("Sensor Data Handler is not running. Cannot provide data.")
             return None

        aggregated = {}
        latest_overall_timestamp = None

        # --- Simple Aggregation and Extraction for Consumers ---
        # This needs refinement based on exact sensor list and AI/Evaluator needs.
        # This example just takes the latest value stored for each sensor.
        # Real synchronization (e.g., values within a 100ms window) is more complex.

        values_for_ai_dict = {} # Use dict first to map sensor_name to value
        values_for_evaluator_dict = {} # Separate structure for evaluator

        # Iterate through the latest data from all sensors
        for (r_id, s_name), data_item in self._latest_data.items():
            if robot_id is None or r_id == robot_id: # Filter by robot_id if specified
                 # Update latest overall timestamp
                 item_ts_str = data_item.get('timestamp')
                 if item_ts_str:
                     try:
                         item_ts = datetime.datetime.fromisoformat(item_ts_str.replace('Z', '+00:00'))
                         if latest_overall_timestamp is None or item_ts > latest_overall_timestamp:
                             latest_overall_timestamp = item_ts
                     except ValueError:
                          logger.warning(f"Could not parse timestamp for {r_id}/{s_name}: {item_ts_str}")


                 # --- Extract data for AI Input ---
                 # Map sensor name to the position in AI input vector (needs config lookup)
                 # This mapping must be consistent with data_handler/trainer/ai_model.
                 # Example mapping based on config.SENSOR_FEATURE_SIZE = 4:
                 ai_input_map = {
                     'welder_current': 0,
                     'welder_voltage': 1,
                     'welder_speed': 2, # Could be wire feed or robot speed
                     'temperature': 3,
                     # Add other sensors mapped to AI features...
                 }
                 if s_name in ai_input_map and data_item.get('value_type') == 'real':
                      values_for_ai_dict[ai_input_map[s_name]] = data_item.get('value')
                 # Handle other value_types for AI input if needed (e.g., image path for vision)
                 # if s_name == 'vision_image' and data_item.get('value_type') == 'path':
                 #      aggregated['image_path_for_ai'] = data_item.get('value')


                 # --- Extract data for Evaluator ---
                 # Evaluator might need different or more direct access to certain sensor values
                 values_for_evaluator_dict[s_name] = data_item.get('value') # Simple direct value
                 # Or the whole data item: values_for_evaluator_dict[s_name] = data_item


        # --- Finalize AI Input Vector ---
        # Ensure the AI input vector has the correct size and order, filling missing values if necessary
        values_for_ai_list = [None] * config.SENSOR_FEATURE_SIZE # Initialize with None or a default
        essential_ai_sensors_missing = False
        # Check if we have data for all required AI features
        for idx in range(config.SENSOR_FEATURE_SIZE):
             if idx in values_for_ai_dict:
                  values_for_ai_list[idx] = values_for_ai_dict[idx]
             else:
                  # This sensor data required for AI is missing in the latest batch!
                  # Decide how to handle: use previous value, use default, mark as missing/error
                  # For this example, we check if *any* essential data is missing
                  essential_ai_sensors_missing = True
                  logger.warning(f"Sensor data for AI feature index {idx} is missing in latest data for robot {robot_id}.")
                  # You might fill with a placeholder like 0.0 or the last known value:
                  # values_for_ai_list[idx] = 0.0 # Or self._last_known_ai_values.get((robot_id, idx), 0.0)

        # Decide if we return data if essential sensors are missing
        if essential_ai_sensors_missing:
             # Option 1: Return None or error indicator
             # return None # Or {'error': 'Missing essential sensor data for AI'}
             # Option 2: Return with placeholders (less robust)
             logger.warning("Returning AI data with missing essential sensor values.")
             # Convert list to numpy array for AI engine
             aggregated['values_for_ai'] = np.array(values_for_ai_list, dtype=np.float32)
        else:
             # Convert list to numpy array for AI engine
             aggregated['values_for_ai'] = np.array(values_for_ai_list, dtype=np.float32)


        # --- Finalize Aggregated Output ---
        aggregated['robot_id'] = robot_id # Add robot_id to output if filtered
        aggregated['timestamp'] = latest_overall_timestamp.isoformat() if latest_overall_timestamp else None
        aggregated['values_for_evaluator'] = values_for_evaluator_dict
        aggregated['source'] = 'sensor_handler' # Indicate data origin

        # Return None if no data was collected at all for the requested robot
        if not aggregated.get('values_for_ai') and not aggregated.get('values_for_evaluator'):
             return None # No data available

        # logger.debug(f"Aggregated data for robot {robot_id}: {aggregated}")
        return aggregated

    def get_status(self):
        """Returns the current status of the sensor data handler."""
        collector_statuses = {c.sensor_name: c._running for c in self._collectors}
        return {
            "handler_running": self._running,
            "processing_thread_alive": self._processing_thread.is_alive() if self._processing_thread else False,
            "data_queue_size": self._data_queue.qsize(),
            "latest_aggregated_data_count": len(self._latest_data), # Number of sensors with latest data stored
            "collector_statuses": collector_statuses
        }


# Example Usage (requires DummyConfig and DummyDataLoggerDB)
if __name__ == '__main__':
    logger.info("--- Sensor Data Handler Example ---")

    # --- Dummy Config (for example purposes only) ---
    class DummyConfig:
        SENSOR_CONFIGS = [
            {'name': 'welder_current', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.05},
            {'name': 'welder_voltage', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.06},
            {'name': 'welder_speed',   'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.07},
            {'name': 'temperature',    'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.5}, # Slower sensor
            {'name': 'arc_stability_index', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.1},
            # Add sensor for Robot 2
             {'name': 'welder_current', 'protocol': 'dummy', 'robot_id': 2, 'interval_sec': 0.05},
             {'name': 'temperature',    'protocol': 'dummy', 'robot_id': 2, 'interval_sec': 0.5},
             # Add image sensor (requires data/images/ directory to exist for dummy path)
             {'name': 'vision_image', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 1.0},
        ]
        # Define what constitutes the AI input features in order (must match ai_model/trainer)
        # This mapping is used in get_latest_aggregated_data
        # This is a conceptual config, manually handled in get_latest_aggregated_data for now.
        SENSOR_FEATURE_SIZE = 4 # Example: current, voltage, speed, temperature
        # Assuming the order is [welder_current, welder_voltage, welder_speed, temperature]

        # Add DATABASE_PATH or other config for DataLoggerDB if needed

    config = DummyConfig()

    # --- Dummy DataLoggerDB (for example purposes) ---
    class DummyDataLoggerDB:
        def log_sensor_reading(self, sensor_name, sensor_value, robot_id=None, job_id=None, extra_info=None):
            # print(f"LOG (Sensor): Robot {robot_id} | {sensor_name}: {sensor_value} | Extra: {extra_info}")
            pass # Suppress logging in example for clarity

    dummy_logger = DummyDataLoggerDB()


    # Ensure dummy image directory exists if using vision_image sensor config
    if 'vision_image' in [cfg['name'] for cfg in config.SENSOR_CONFIGS]:
        dummy_image_dir = 'data/images/'
        if not os.path.exists(dummy_image_dir):
             os.makedirs(dummy_image_dir)
             logger.info(f"Created dummy image directory: {dummy_image_dir}")


    # --- Initialize the Sensor Data Handler ---
    sensor_handler = SensorDataHandler(data_logger=dummy_logger)


    # --- Start Collection ---
    sensor_handler.start_collection()

    # --- Monitor and Get Data ---
    logger.info("\nMonitoring sensor data for 5 seconds...")
    for i in range(10): # Loop for 10 cycles (5 seconds total with 0.5s sleep)
        status = sensor_handler.get_status()
        logger.info(f"Handler Status (Cycle {i+1}): {status}")

        # Get latest aggregated data for Robot 1 (AI input + Evaluator info)
        aggregated_data_r1 = sensor_handler.get_latest_aggregated_data(robot_id=1)
        if aggregated_data_r1:
            # Simulate passing this data to AI Inference Engine and Quality Evaluator
            # logger.info(f"Latest Aggregated Data for Robot 1: {aggregated_data_r1}")
            # Example: ai_engine.process_sensor_data(aggregated_data_r1['values_for_ai'])
            # Example: quality_controller.evaluate_quality(..., aggregated_data_r1['values_for_evaluator'])
            logger.info(f"Robot 1 Latest AI Input Data: {aggregated_data_r1.get('values_for_ai')}")
            logger.info(f"Robot 1 Latest Evaluator Data (Temp): {aggregated_data_r1.get('values_for_evaluator', {}).get('temperature')}") # Example access
            logger.info(f"Robot 1 Latest Evaluator Data (Image Path): {aggregated_data_r1.get('values_for_evaluator', {}).get('vision_image')}") # Example access


        # Get latest aggregated data for Robot 2
        aggregated_data_r2 = sensor_handler.get_latest_aggregated_data(robot_id=2)
        if aggregated_data_r2:
            logger.info(f"Robot 2 Latest AI Input Data: {aggregated_data_r2.get('values_for_ai')}") # Needs full set of features for AI

        time.sleep(0.5) # Sleep between checks

    # --- Stop Collection ---
    sensor_handler.stop_collection()

    # --- Cleanup ---
    # Remove dummy image directory if created
    if 'vision_image' in [cfg['name'] for cfg in config.SENSOR_CONFIGS] and os.path.exists(dummy_image_dir):
         import shutil
         logger.info(f"Removing dummy image directory: {dummy_image_dir}")
         # shutil.rmtree(dummy_image_dir) # Use with caution!
         # For this simple example, just remove the file if it exists
         # Note: Dummy file path includes timestamp, hard to remove specifically.
         # A real cleanup needs to track created dummy files.


    logger.info("--- Sensor Data Handler Example Finished ---")