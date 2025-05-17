# src/sensor_data_handler.py
# Description: (MODIFIED) Handles real-time collection and preprocessing of sensor data.
#              Publishes sensor data to a message queue for consumption by other modules.

import time
import threading
import logging
import queue # For internal buffering from collectors before MQ publishing
import random
import datetime
import json
import os

# Import Message Queue client (e.g., pika for RabbitMQ)
# Ensure 'pika' is in requirements.txt if using RabbitMQ
try:
    import pika
except ImportError:
    pika = None # Mark as unavailable

from src import config # Import the main config module
# DataLoggerDB can be used here for local logging of collection events if needed,
# but primary sensor data logging should happen via a subscriber to the MQ.
# from src.data_logger_db import DataLoggerDB

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- Sensor Data Format for Message Queue (Example) ---
# {
#   "timestamp_utc": "ISO8601_UTC_string", // UTC timestamp for the reading
#   "timestamp_device": "ISO8601_string_optional", // Original timestamp from device if available
#   "robot_id": 1, // Associated robot ID
#   "sensor_id": "welder_current_robot1", // Unique sensor identifier
#   "sensor_name": "welder_current", // Generic sensor name/type
#   "value_type": "real" | "integer" | "text" | "json_object" | "image_path" | "binary_base64",
#   "value": 150.5, // Actual sensor value
#   "unit": "Amperes", // Optional unit
#   "metadata": { ... } // Optional additional metadata
# }


class SensorCollector:
    """
    Manages data collection from a single sensor source (or simulates it).
    Puts collected data into a shared queue for the SensorDataHandler to process.
    """
    def __init__(self, sensor_config, output_queue, data_logger=None):
        self.sensor_config = sensor_config
        self.sensor_id = sensor_config.get('id', f"{sensor_config.get('name', 'unknown')}_{sensor_config.get('robot_id', 'global')}") # Unique ID
        self.sensor_name = sensor_config.get('name', 'unknown_sensor')
        self.robot_id = sensor_config.get('robot_id', None)
        self.data_logger = data_logger # Optional: For logging collector-specific events

        self._output_queue = output_queue # Shared queue to send data to SensorDataHandler's processing loop
        self._running = False
        self._collection_thread = None
        self._stop_event = threading.Event()

        # --- Sensor Connection (Placeholder for actual connection logic) ---
        self._sensor_connection = None # Store actual connection object (socket, serial port, API client)
        self._is_sensor_connected = False

        logger.info(f"Sensor Collector '{self.sensor_id}' initialized.")

    def _connect_to_sensor(self):
        """Placeholder for establishing connection to the actual sensor hardware/source."""
        protocol = self.sensor_config.get('protocol', 'dummy')
        if protocol == 'dummy':
            logger.debug(f"Sensor '{self.sensor_id}': Using dummy protocol, no real connection needed.")
            self._is_sensor_connected = True
            return True
        elif protocol == 'tcp':
            # Example TCP connection logic (needs host, port in sensor_config)
            # host = self.sensor_config.get('host')
            # port = self.sensor_config.get('port')
            # try:
            #     self._sensor_connection = socket.create_connection((host, port), timeout=2)
            #     self._is_sensor_connected = True
            #     logger.info(f"Sensor '{self.sensor_id}': Connected via TCP to {host}:{port}")
            #     return True
            # except Exception as e:
            #     logger.error(f"Sensor '{self.sensor_id}': Failed to connect TCP sensor {host}:{port} - {e}")
            #     self._is_sensor_connected = False
            #     return False
            logger.warning(f"Sensor '{self.sensor_id}': TCP protocol not fully implemented in this example.")
            self._is_sensor_connected = False # Simulate connection failure
            return False # TCP connection logic needs to be implemented
        # Add other protocols: 'udp', 'serial', 'http_api', 'opc_ua', 'modbus' etc.
        else:
            logger.warning(f"Sensor '{self.sensor_id}': Unknown protocol '{protocol}'. Cannot connect.")
            self._is_sensor_connected = False
            return False

    def _disconnect_from_sensor(self):
        """Placeholder for closing connection to the sensor hardware/source."""
        if self._sensor_connection:
            try:
                # Example: if self._sensor_connection is a socket
                # self._sensor_connection.close()
                pass
            except Exception as e:
                logger.error(f"Sensor '{self.sensor_id}': Error disconnecting sensor: {e}")
        self._sensor_connection = None
        self._is_sensor_connected = False
        logger.info(f"Sensor '{self.sensor_id}': Disconnected (placeholder).")


    def start(self):
        """Starts the data collection thread for this sensor."""
        if self._running:
            logger.warning(f"Sensor Collector '{self.sensor_id}' is already running.")
            return
        self._running = True
        self._stop_event.clear()
        self._collection_thread = threading.Thread(target=self._collect_loop, name=f"Collector-{self.sensor_id}", daemon=True)
        self._collection_thread.start()
        logger.info(f"Sensor Collector '{self.sensor_id}' thread started.")

    def stop(self):
        """Stops the data collection thread."""
        if self._running:
            logger.info(f"Sensor Collector '{self.sensor_id}' stopping...")
            self._running = False
            self._stop_event.set() # Signal the loop to stop
            if self._collection_thread and self._collection_thread.is_alive():
                self._collection_thread.join(timeout=5) # Wait for thread to finish
                if self._collection_thread.is_alive():
                    logger.warning(f"Sensor Collector '{self.sensor_id}' thread did not terminate gracefully.")
            self._disconnect_from_sensor() # Ensure sensor connection is closed
            logger.info(f"Sensor Collector '{self.sensor_id}' stopped.")
        self._collection_thread = None


    def _collect_loop(self):
        """The main loop for collecting data from a single sensor."""
        collection_interval = self.sensor_config.get('interval_sec', 0.1)

        while self._running and not self._stop_event.is_set():
            if not self._is_sensor_connected:
                if not self._connect_to_sensor():
                    # Connection failed, wait before retrying
                    self._stop_event.wait(timeout=5.0) # Wait 5 seconds before retry
                    continue # Retry connection

            loop_start_time = time.time()
            collected_data_packet = None

            try:
                # --- Actual Data Collection from Sensor (Based on Protocol) ---
                # This part needs to be implemented for each specific sensor protocol.
                protocol = self.sensor_config.get('protocol', 'dummy')
                if protocol == 'dummy':
                    # Generate dummy data (same as previous SensorDataHandler example)
                    value, value_type, unit, metadata = self._generate_dummy_value()
                    if value is not None:
                         collected_data_packet = {
                             "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "robot_id": self.robot_id,
                             "sensor_id": self.sensor_id,
                             "sensor_name": self.sensor_name,
                             "value_type": value_type,
                             "value": value,
                             "unit": unit,
                             "metadata": metadata
                         }
                # elif protocol == 'tcp' and self._sensor_connection:
                #     # raw_data = self._sensor_connection.recv(1024)
                #     # collected_data_packet = self._parse_tcp_data(raw_data) # Implement this
                #     pass
                else:
                    logger.warning(f"Sensor '{self.sensor_id}': Data collection for protocol '{protocol}' not implemented.")
                    # Simulate periodic failure or no data for non-dummy protocols
                    self._disconnect_from_sensor() # Assume connection issue
                    self._stop_event.wait(timeout=collection_interval) # Wait before next attempt
                    continue


                # --- Basic Preprocessing (Optional, can be done in main handler) ---
                if collected_data_packet:
                    # Example: Validate value ranges, convert units, filter noise
                    pass # Add preprocessing steps if needed


                # --- Enqueue Data for Main Handler ---
                if collected_data_packet:
                    try:
                        self._output_queue.put(collected_data_packet, timeout=0.5) # Put with timeout
                        # logger.debug(f"Sensor '{self.sensor_id}': Enqueued data.") # Too verbose
                    except queue.Full:
                        logger.warning(f"Sensor '{self.sensor_id}': Output queue is full. Data might be lost.")
                        # Implement strategy for full queue (e.g., drop oldest, wait longer)

                    # Optional: Log raw collection event locally via DataLoggerDB if passed
                    # if self.data_logger:
                    #     self.data_logger.log_sensor_reading(...) # This would be direct DB write

            except Exception as e:
                logger.error(f"Sensor Collector '{self.sensor_id}': Error during collection cycle: {e}", exc_info=True)
                self._is_sensor_connected = False # Assume connection lost on error
                # Consider specific error handling (e.g., reconnect attempts within this loop)

            # Control loop frequency
            elapsed_time = time.time() - loop_start_time
            sleep_time = collection_interval - elapsed_time
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time) # Wait or until stop is signaled
        logger.info(f"Sensor Collector '{self.sensor_id}': Loop finished.")


    def _generate_dummy_value(self):
        """Generates a dummy value based on sensor name (for dummy protocol)."""
        value, value_type, unit, metadata = None, 'unknown', None, {}
        s_name = self.sensor_name.lower()
        if 'current' in s_name:
            value, value_type, unit = random.uniform(100, 200), 'real', 'Amperes'
        elif 'voltage' in s_name:
            value, value_type, unit = random.uniform(18, 28), 'real', 'Volts'
        elif 'speed' in s_name or 'wire_feed' in s_name:
            value, value_type, unit = random.uniform(200, 400), 'real', 'mm/min' # Or m/min, etc.
        elif 'temperature' in s_name:
            value, value_type, unit = random.uniform(250, 450), 'real', 'Celsius'
        elif 'arc_stability' in s_name:
            value, value_type, unit = random.uniform(0.5, 0.95), 'real', 'index'
        elif 'vision_image' in s_name:
            ts_file_part = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            value = os.path.join(config.IMAGE_DATA_DIR, f"robot{self.robot_id}_{self.sensor_id}_{ts_file_part}.jpg")
            value_type = 'image_path'
            metadata = {'resolution': '640x480', 'format': 'jpg'}
            # Simulate creating a dummy image file
            # try:
            #     with open(value, 'w') as f: f.write("dummy image data")
            # except Exception: pass # Ignore if cannot write in example
        else: # Generic sensor
            value, value_type = random.randint(0,100) if random.random() > 0.5 else random.random()*100, 'integer' if isinstance(value, int) else 'real'

        return value, value_type, unit, metadata


class SensorDataHandler:
    """
    Manages multiple SensorCollector instances, aggregates/distributes data
    by publishing to a Message Queue (e.g., RabbitMQ).
    """
    _instance = None
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

             logger.info("Sensor Data Handler (Advanced) initializing...")
             self.data_logger = data_logger

             self.sensor_configs = getattr(config, 'SENSOR_CONFIGS', [])
             if not self.sensor_configs:
                 logger.warning("No SENSOR_CONFIGS found in config.py. Sensor data collection will be non-functional.")

             self._collected_data_queue = queue.Queue(maxsize=1000) # Queue for data from collectors
             self._collectors = []
             self._publisher_thread = None # Thread for publishing data from queue to MQ
             self._running = False
             self._stop_event = threading.Event()

             # --- Message Queue (RabbitMQ) Setup ---
             self.mq_connection = None
             self.mq_channel = None
             self._mq_lock = threading.Lock() # For thread-safe MQ operations
             if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
                 if not pika:
                     logger.error("Pika library not found for RabbitMQ. Message queue publishing will be disabled.")
                 else:
                     self._connect_mq() # Attempt initial MQ connection
             elif config.USE_MESSAGE_QUEUE:
                 logger.warning(f"Message Queue type '{config.MQ_TYPE}' not yet supported in this example. Publishing disabled.")


             # Create SensorCollector instances
             for sensor_cfg in self.sensor_configs:
                 try:
                     collector = SensorCollector(sensor_cfg, self._collected_data_queue, self.data_logger)
                     self._collectors.append(collector)
                 except Exception as e_collector:
                     logger.error(f"Failed to initialize collector for config {sensor_cfg.get('id', 'N/A')}: {e_collector}")

             logger.info(f"Sensor Data Handler initialized with {len(self._collectors)} collectors.")


    def _connect_mq(self):
        """Establishes connection to RabbitMQ and declares exchange."""
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika:
            return False
        with self._mq_lock:
            if self.mq_channel and self.mq_channel.is_open:
                return True
            try:
                logger.info(f"Connecting to RabbitMQ at {config.MQ_HOST}:{config.MQ_PORT}")
                credentials = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                parameters = pika.ConnectionParameters(
                    host=config.MQ_HOST,
                    port=config.MQ_PORT,
                    virtual_host=config.MQ_VHOST,
                    credentials=credentials,
                    heartbeat=600, # Enable heartbeat
                    blocked_connection_timeout=300
                )
                self.mq_connection = pika.BlockingConnection(parameters)
                self.mq_channel = self.mq_connection.channel()
                # Declare an exchange (e.g., a topic exchange for sensor data)
                self.mq_channel.exchange_declare(exchange=config.MQ_SENSOR_DATA_EXCHANGE, exchange_type='topic', durable=True)
                logger.info(f"RabbitMQ connection established and exchange '{config.MQ_SENSOR_DATA_EXCHANGE}' declared.")
                # Optional: Add connection/channel error callbacks for robustness
                # self.mq_connection.add_on_connection_blocked_callback(...)
                # self.mq_connection.add_on_connection_unblocked_callback(...)
                return True
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}", exc_info=True)
                if self.mq_connection and self.mq_connection.is_open: self.mq_connection.close()
                self.mq_connection = None
                self.mq_channel = None
                return False

    def _disconnect_mq(self):
        """Closes RabbitMQ connection."""
        with self._mq_lock:
            if self.mq_channel and self.mq_channel.is_open:
                try: self.mq_channel.close()
                except Exception as e_ch: logger.error(f"Error closing MQ channel: {e_ch}")
            if self.mq_connection and self.mq_connection.is_open:
                try: self.mq_connection.close()
                except Exception as e_conn: logger.error(f"Error closing MQ connection: {e_conn}")
            self.mq_channel = None
            self.mq_connection = None
            logger.info("RabbitMQ connection closed.")


    def _publish_to_mq(self, routing_key, message_body_dict):
        """Publishes a message to the configured RabbitMQ exchange."""
        if not self.mq_channel or not self.mq_channel.is_open:
            logger.warning(f"MQ channel not open. Attempting to reconnect to MQ for publishing.")
            if not self._connect_mq(): # Try to reconnect
                logger.error(f"Failed to publish to MQ, connection error. Message for {routing_key} might be lost.")
                return False

        try:
            message_body_str = json.dumps(message_body_dict, ensure_ascii=False)
            with self._mq_lock: # Protect channel operations
                self.mq_channel.basic_publish(
                    exchange=config.MQ_SENSOR_DATA_EXCHANGE,
                    routing_key=routing_key, # e.g., "sensor.raw.robot1.temperature"
                    body=message_body_str,
                    properties=pika.BasicProperties(
                        delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE, # Make message persistent
                        content_type='application/json',
                        timestamp=int(time.time()) # UNIX timestamp
                    )
                )
            # logger.debug(f"Published to MQ: Key='{routing_key}', Body='{message_body_str[:100]}...'") # Verbose
            return True
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker, socket.error) as e_pub:
             logger.error(f"MQ Publishing error (connection issue): {e_pub}. Attempting reconnect.")
             self._disconnect_mq() # Force close and try to reconnect in next cycle
             # Consider re-queueing the message or saving to a dead-letter queue
             return False
        except Exception as e:
            logger.error(f"Failed to publish message to MQ. RoutingKey: {routing_key}. Error: {e}", exc_info=True)
            return False


    def start_collection(self):
        """Starts all sensor collector threads and the MQ publisher thread."""
        if self._running:
            logger.warning("Sensor Data Handler collection is already running.")
            return

        logger.info("Starting sensor data collection and MQ publishing...")
        self._running = True
        self._stop_event.clear()

        # Ensure MQ is connected before starting publisher thread
        if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
            if not (self.mq_channel and self.mq_channel.is_open):
                self._connect_mq() # Attempt to connect MQ if not already

        # Start individual sensor collectors
        for collector in self._collectors:
            collector.start()

        # Start the MQ publisher thread
        if config.USE_MESSAGE_QUEUE:
            self._publisher_thread = threading.Thread(target=self._mq_publisher_loop, name="MQPublisher", daemon=True)
            self._publisher_thread.start()

        logger.info("Sensor data collection and MQ publishing started.")


    def stop_collection(self):
        """Stops all sensor collector threads and the MQ publisher thread."""
        if not self._running:
            logger.info("Sensor Data Handler collection is not running.")
            return

        logger.info("Stopping sensor data collection and MQ publishing...")
        self._running = False
        self._stop_event.set() # Signal publisher thread to stop

        # Stop individual sensor collectors
        for collector in self._collectors:
            collector.stop() # This method now waits for its thread to join

        # Wait for the MQ publisher thread to finish
        if self._publisher_thread and self._publisher_thread.is_alive():
            logger.info("Waiting for MQ publisher thread to complete...")
            self._publisher_thread.join(timeout=config.DB_LOG_BATCH_TIMEOUT_SEC + 2.0) # Use a similar timeout
            if self._publisher_thread.is_alive():
                logger.warning("MQ publisher thread did not terminate gracefully.")
        self._publisher_thread = None

        # Disconnect from Message Queue
        if config.USE_MESSAGE_QUEUE:
             self._disconnect_mq()

        logger.info("Sensor data collection and MQ publishing stopped.")


    def _mq_publisher_loop(self):
        """Internal thread loop to get data from _collected_data_queue and publish to MQ."""
        logger.info("MQ Publisher thread started. Waiting for sensor data...")
        while not self._stop_event.is_set():
            try:
                # Get data from the queue (with a timeout so we can check stop_event)
                sensor_data_packet = self._collected_data_queue.get(timeout=1.0) # Wait 1 sec for data

                # Construct routing key (example: sensor.raw.robot1.temperature)
                robot_id_part = f"robot{sensor_data_packet['robot_id']}" if sensor_data_packet['robot_id'] is not None else "global"
                sensor_name_part = sensor_data_packet['sensor_name'].lower().replace(" ", "_")
                routing_key = f"{config.MQ_SENSOR_DATA_ROUTING_KEY_PREFIX}{robot_id_part}.{sensor_name_part}"

                # Publish the sensor data packet to MQ
                if not self._publish_to_mq(routing_key, sensor_data_packet):
                    # Publishing failed, re-queue the data for later attempt (simple retry)
                    # A more robust system might use a dead-letter queue or limited retries.
                    logger.warning(f"Re-queueing sensor data for {sensor_data_packet['sensor_id']} due to MQ publish failure.")
                    try:
                        self._collected_data_queue.put(sensor_data_packet, timeout=0.1) # Try to put back with timeout
                    except queue.Full:
                         logger.error("Failed to re-queue sensor data, internal queue full. Data lost.")

                self._collected_data_queue.task_done()

            except queue.Empty:
                # Queue was empty within the timeout, continue loop and check stop_event
                pass
            except Exception as e:
                logger.error(f"MQ Publisher thread error: {e}", exc_info=True)
                # Sleep briefly after an error before retrying
                time.sleep(0.5)

        # Process any remaining items in the queue upon stopping
        logger.info("MQ Publisher: Stop event received. Processing remaining queue items before exit...")
        while not self._collected_data_queue.empty():
            try:
                sensor_data_packet = self._collected_data_queue.get_nowait()
                # Construct routing key as above
                robot_id_part = f"robot{sensor_data_packet['robot_id']}" if sensor_data_packet['robot_id'] is not None else "global"
                sensor_name_part = sensor_data_packet['sensor_name'].lower().replace(" ", "_")
                routing_key = f"{config.MQ_SENSOR_DATA_ROUTING_KEY_PREFIX}{robot_id_part}.{sensor_name_part}"
                self._publish_to_mq(routing_key, sensor_data_packet) # Attempt to publish remaining
                self._collected_data_queue.task_done()
            except queue.Empty:
                break
            except Exception as e_final:
                 logger.error(f"Error publishing remaining item from queue: {e_final}")
        logger.info("MQ Publisher thread finished.")


    def get_latest_aggregated_data(self, robot_id=None):
        """
        (DEPRECATED or MODIFIED ROLE) Retrieves the latest received data.
        In a MQ-based system, consumers subscribe to topics. This method might provide
        a snapshot from an internal cache for diagnostics or simple HMI, but
        primary data flow is via MQ.
        For now, this can return data from the last published messages or a local cache if implemented.
        """
        logger.warning("get_latest_aggregated_data() called. In an MQ system, consumers should subscribe. This method provides a limited snapshot/is for diagnostics.")
        # This method needs to be re-thought in an MQ architecture.
        # It could query a short-term cache that the publisher also updates,
        # or it might be removed if all consumers use MQ.
        # For this example, let's return a placeholder or data from a conceptual cache.
        # Conceptual: if a local cache `self._latest_published_data` was maintained by _mq_publisher_loop:
        #   return self._latest_published_data.get(robot_id, {}) # Example
        return {"warning": "Data should be consumed from Message Queue.", "robot_id_requested": robot_id}


    def get_status(self):
        """Returns the current status of the sensor data handler."""
        collector_statuses = {c.sensor_id: {"running": c._running, "connected": c._is_sensor_connected} for c in self._collectors}
        return {
            "handler_running": self._running,
            "publisher_thread_alive": self._publisher_thread.is_alive() if self._publisher_thread else False,
            "collected_data_queue_size": self._collected_data_queue.qsize(),
            "mq_connected": self.mq_channel.is_open if self.mq_channel else False,
            "collector_statuses": collector_statuses
        }


# Example Usage (requires DummyConfig and a running RabbitMQ server for MQ parts)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    logger.info("--- Sensor Data Handler (Advanced with MQ) Example ---")

    # Dummy Config for MQ and Sensors
    class DummyConfigAdvanced:
        # MQ settings (ensure RabbitMQ is running on localhost:5672 with guest:guest for this default)
        USE_MESSAGE_QUEUE = True
        MQ_TYPE = 'rabbitmq'
        MQ_HOST = 'localhost'
        MQ_PORT = 5672
        MQ_USER = 'guest'
        MQ_PASSWORD = 'guest'
        MQ_VHOST = '/'
        MQ_SENSOR_DATA_EXCHANGE = 'sensor_data_exchange_test'
        MQ_SENSOR_DATA_ROUTING_KEY_PREFIX = 'sensor.test.'

        SENSOR_CONFIGS = [
            {'id': 'temp_sensor_R1', 'name': 'temperature', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.5},
            {'id': 'current_sensor_R1', 'name': 'welder_current', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 0.1},
            {'id': 'vision_R1', 'name': 'vision_image', 'protocol': 'dummy', 'robot_id': 1, 'interval_sec': 1.0, 'path_prefix': 'data/images_test/'},
        ]
        IMAGE_DATA_DIR = 'data/images_test/' # For dummy image path generation
        # Dummy DataLoggerDB for SensorCollector if needed
        # DATABASE_PATH = 'test_sensor_handler_log.db'

    config = DummyConfigAdvanced()

    # Ensure IMAGE_DATA_DIR exists for dummy image paths
    if not os.path.exists(config.IMAGE_DATA_DIR):
        os.makedirs(config.IMAGE_DATA_DIR, exist_ok=True)


    # --- MQ Consumer (Example, run in a separate script/thread for real testing) ---
    mq_consumer_messages = []
    consumer_stop_event = threading.Event()
    def mq_consumer_thread_func():
        if not pika: return
        logger.info("[MQ Consumer] Starting...")
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=config.MQ_HOST))
            channel = connection.channel()
            channel.exchange_declare(exchange=config.MQ_SENSOR_DATA_EXCHANGE, exchange_type='topic', durable=True)
            result = channel.queue_declare(queue='', exclusive=True) # Anonymous, exclusive queue
            queue_name = result.method.queue
            # Bind to all sensor data for this test
            channel.queue_bind(exchange=config.MQ_SENSOR_DATA_EXCHANGE, queue=queue_name, routing_key=f"{config.MQ_SENSOR_DATA_ROUTING_KEY_PREFIX}#")
            logger.info(f"[MQ Consumer] Waiting for messages on queue '{queue_name}'. To exit press CTRL+C")

            def callback(ch, method, properties, body):
                logger.info(f"[MQ Consumer] Received [x] {method.routing_key}:{body.decode()[:100]}...")
                mq_consumer_messages.append(json.loads(body.decode()))
                # ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge message

            channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

            # Keep consuming until stop_event is set
            while not consumer_stop_event.is_set():
                connection.process_data_events(time_limit=0.1) # Process events with a timeout
                if not connection.is_open: break
            logger.info("[MQ Consumer] Stopping consumption...")
            channel.stop_consuming() # This might not be enough if connection.process_data_events is blocking
            connection.close()

        except pika.exceptions.AMQPConnectionError as e_conn:
             logger.error(f"[MQ Consumer] Connection error: {e_conn}. Ensure RabbitMQ is running.")
        except Exception as e:
            logger.error(f"[MQ Consumer] Error: {e}", exc_info=True)
        logger.info("[MQ Consumer] Stopped.")


    # Start MQ consumer in a thread for this example (if pika is available)
    consumer_thread = None
    if pika and config.USE_MESSAGE_QUEUE:
        consumer_thread = threading.Thread(target=mq_consumer_thread_func, name="MQTestConsumer", daemon=True)
        consumer_thread.start()
        time.sleep(1) # Give consumer a moment to start up and bind


    # --- Initialize and Test SensorDataHandler ---
    # Assume a dummy logger for this example if needed by SensorCollector
    # dummy_data_logger = DataLoggerDB() # Get singleton
    sensor_handler = SensorDataHandler(data_logger=None) # Pass None if direct logging in collector is disabled

    try:
        if config.USE_MESSAGE_QUEUE and not (sensor_handler.mq_channel and sensor_handler.mq_channel.is_open):
            logger.warning("MQ not connected at handler init. Example might not publish.")

        sensor_handler.start_collection()
        logger.info("\nSensor collection started. Monitoring for 5 seconds...")

        for i in range(10): # Check status for 5 seconds
            status = sensor_handler.get_status()
            logger.info(f"Handler Status (Cycle {i+1}): {status}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    except Exception as e:
        logger.error(f"Error in example main: {e}", exc_info=True)
    finally:
        logger.info("\nStopping sensor collection...")
        sensor_handler.stop_collection()

        if consumer_thread:
            logger.info("Stopping MQ consumer thread...")
            consumer_stop_event.set()
            consumer_thread.join(timeout=5)
            if consumer_thread.is_alive(): logger.warning("MQ consumer thread did not join.")

        logger.info(f"\nTotal messages received by MQ consumer: {len(mq_consumer_messages)}")
        # for msg in mq_consumer_messages[:5]: # Print first 5 received messages
        #     print(f"  Consumed: {msg}")

        # Cleanup dummy image dir
        if os.path.exists(config.IMAGE_DATA_DIR):
             import shutil
             try:
                 # shutil.rmtree(config.IMAGE_DATA_DIR) # Careful with rmtree
                 # logger.info(f"Cleaned up dummy image directory: {config.IMAGE_DATA_DIR}")
                 pass # Avoid accidental deletion in general examples
             except Exception as e_clean:
                 logger.error(f"Error cleaning up image directory: {e_clean}")


    logger.info("--- Sensor Data Handler (Advanced with MQ) Example Finished ---")