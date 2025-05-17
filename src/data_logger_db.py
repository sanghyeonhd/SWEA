# src/data_logger_db.py
# Description: (MODIFIED) Handles logging of all system data to a production-level database (e.g., PostgreSQL).
#              Implements batch logging for performance.

import json
import datetime
import logging
import threading
import queue
import time

# Import specific DB connector based on config.DB_TYPE
# For PostgreSQL, ensure 'psycopg2-binary' is in requirements.txt
try:
    import psycopg2
    import psycopg2.extras # For extras like RealDictCursor
except ImportError:
    psycopg2 = None # Mark as unavailable if not installed

# For SQLite (fallback or specific use)
import sqlite3

from src import config # Import the main config module

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- Database Schema Notes (Conceptual for PostgreSQL - needs exact DDL) ---
# Table: process_events (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, job_id TEXT, robot_id INTEGER, event_type TEXT NOT NULL, details JSONB)
# Table: sensor_readings (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, robot_id INTEGER, sensor_name TEXT NOT NULL, sensor_value REAL, sensor_extra_info JSONB, job_id TEXT)
# Table: ai_predictions (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, job_id TEXT, robot_id INTEGER, input_data JSONB, predicted_class INTEGER, probabilities JSONB, confidence REAL, predicted_score REAL, raw_output JSONB)
# Table: quality_evaluations (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, job_id TEXT, robot_id INTEGER, ai_prediction_id INTEGER REFERENCES ai_predictions(id), sensor_data_summary JSONB, combined_status TEXT, detected_issues JSONB, numeric_score REAL)
# Table: adaptive_control_actions (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, job_id TEXT, robot_id INTEGER, triggering_evaluation_id INTEGER REFERENCES quality_evaluations(id), suggested_adjustments JSONB, action_taken_status TEXT, actual_applied_params JSONB)
# Table: robot_status_logs (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, robot_id INTEGER NOT NULL, status_data JSONB, job_id TEXT)
# --- Additional tables might include: welding_recipes, materials_info, operator_logs, etc.

class DataLoggerDB:
    """
    Manages database connections and provides methods to log various system data.
    Supports batch logging to improve performance.
    """
    _instance = None
    _lock = threading.Lock() # For thread-safe singleton instantiation

    # Batch logging parameters (configurable via config.py if needed)
    BATCH_SIZE = getattr(config, 'DB_LOG_BATCH_SIZE', 100) # Number of logs to batch before writing
    BATCH_TIMEOUT_SEC = getattr(config, 'DB_LOG_BATCH_TIMEOUT_SEC', 5.0) # Max time to wait before writing batch

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            self.db_type = config.DB_TYPE
            self.conn = None
            self.db_path = None # For SQLite
            self._log_queue = queue.Queue(maxsize=self.BATCH_SIZE * 5) # Queue for pending log entries
            self._batch_writer_thread = None
            self._stop_writer_event = threading.Event()
            self._db_access_lock = threading.Lock() # Protects self.conn and cursor operations

            logger.info(f"DataLoggerDB initializing with DB_TYPE: {self.db_type}, Batch Size: {self.BATCH_SIZE}, Batch Timeout: {self.BATCH_TIMEOUT_SEC}s")
            self._connect()
            if self.conn:
                 # Create tables only if using SQLite and they don't exist.
                 # For PostgreSQL, schema management is typically done externally (e.g., migrations).
                 if self.db_type == 'sqlite':
                     self._create_sqlite_tables_if_not_exist() # Example for SQLite
                 # Start the batch writer thread
                 self._start_batch_writer()
            else:
                 logger.error("DataLoggerDB failed to connect to database. Logging will be disabled.")


    def _connect(self):
        """Establishes a database connection based on config.DB_TYPE."""
        with self._db_access_lock: # Protect connection attempt
            if self.conn: # Already connected
                return

            try:
                if self.db_type == 'postgresql':
                    if not psycopg2:
                        logger.error("psycopg2 library not found. Please install it for PostgreSQL support.")
                        return
                    self.conn = psycopg2.connect(
                        dbname=config.DB_NAME,
                        user=config.DB_USER,
                        password=config.DB_PASSWORD,
                        host=config.DB_HOST,
                        port=config.DB_PORT
                    )
                    logger.info(f"Successfully connected to PostgreSQL database: {config.DB_NAME} at {config.DB_HOST}")
                elif self.db_type == 'sqlite':
                    self.db_path = config.SQLITE_DB_PATH
                    # Ensure directory exists for SQLite file
                    os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                    self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # Allow access from writer thread
                    self.conn.row_factory = sqlite3.Row # Access columns by name
                    logger.info(f"Successfully connected to SQLite database: {self.db_path}")
                else:
                    logger.error(f"Unsupported DB_TYPE: {self.db_type}")
                    self.conn = None
            except Exception as e:
                logger.error(f"Error connecting to {self.db_type} database: {e}", exc_info=True)
                self.conn = None

    def _ensure_connection(self):
        """Ensures a database connection is active, reconnects if necessary."""
        # This simple version just checks and calls connect.
        # A more robust version would implement exponential backoff for reconnections.
        with self._db_access_lock:
            if self.conn is None:
                logger.warning("Database connection lost or not established. Attempting to reconnect...")
                self._connect()
            # For PostgreSQL, check if connection is still valid
            if self.db_type == 'postgresql' and self.conn and self.conn.closed != 0:
                logger.warning("PostgreSQL connection was closed. Attempting to reconnect...")
                self._connect()
            return self.conn is not None


    def _create_sqlite_tables_if_not_exist(self):
        """Creates SQLite tables if they do not already exist (for example purposes)."""
        if self.db_type != 'sqlite' or not self.conn:
            return
        # Using simplified schema similar to previous version for SQLite example
        # For PostgreSQL, use proper DDL scripts and migration tools
        queries = [
            """CREATE TABLE IF NOT EXISTS process_events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, job_id TEXT, robot_id INTEGER, event_type TEXT NOT NULL, details TEXT);""",
            """CREATE TABLE IF NOT EXISTS sensor_readings (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, robot_id INTEGER, sensor_name TEXT NOT NULL, sensor_value REAL, sensor_extra_info TEXT, job_id TEXT);""",
            """CREATE TABLE IF NOT EXISTS ai_predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, job_id TEXT, robot_id INTEGER, input_data TEXT, predicted_class INTEGER, probabilities TEXT, confidence REAL, predicted_score REAL, raw_output TEXT);""",
            # Add other tables as needed...
        ]
        try:
            with self.conn: # Context manager handles commit/rollback
                cursor = self.conn.cursor()
                for query in queries:
                    cursor.execute(query)
            logger.info("SQLite tables checked/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating SQLite tables: {e}")


    def _start_batch_writer(self):
        """Starts the background thread for batch writing logs to the database."""
        if self._batch_writer_thread is None or not self._batch_writer_thread.is_alive():
            self._stop_writer_event.clear()
            self._batch_writer_thread = threading.Thread(target=self._batch_writer_loop, name="DBBatchWriter", daemon=True)
            self._batch_writer_thread.start()
            logger.info("Database batch writer thread started.")

    def _batch_writer_loop(self):
        """Continuously polls the log queue and writes batches to the database."""
        logger.info("DBBatchWriter thread waiting for logs...")
        while not self._stop_writer_event.is_set():
            batch = []
            # Try to get items from the queue to form a batch
            try:
                # Wait for the first item with a timeout (BATCH_TIMEOUT_SEC)
                # This allows the loop to wake up periodically even if queue is empty for a while
                log_entry = self._log_queue.get(timeout=self.BATCH_TIMEOUT_SEC)
                batch.append(log_entry)
                self._log_queue.task_done()

                # Try to get more items up to BATCH_SIZE without blocking further
                while len(batch) < self.BATCH_SIZE:
                    try:
                        log_entry = self._log_queue.get_nowait() # Non-blocking get
                        batch.append(log_entry)
                        self._log_queue.task_done()
                    except queue.Empty:
                        break # Queue is empty, process current batch
            except queue.Empty:
                # Timeout occurred, no items in queue for BATCH_TIMEOUT_SEC
                # Continue loop to check stop_event
                continue

            if batch:
                self._write_batch_to_db(batch)

        # Process any remaining items in the queue upon stopping
        logger.info("DBBatchWriter: Stop event received. Processing remaining queue items...")
        remaining_batch = []
        while not self._log_queue.empty():
            try:
                remaining_batch.append(self._log_queue.get_nowait())
                self._log_queue.task_done()
                if len(remaining_batch) >= self.BATCH_SIZE:
                     self._write_batch_to_db(remaining_batch)
                     remaining_batch = []
            except queue.Empty:
                break
        if remaining_batch: # Write any final leftovers
            self._write_batch_to_db(remaining_batch)
        logger.info("DBBatchWriter thread finished.")

    def _write_batch_to_db(self, batch_of_logs):
        """Writes a batch of log entries to the database."""
        if not batch_of_logs:
            return
        if not self._ensure_connection(): # Ensure connection before writing
            logger.error(f"Cannot write batch of {len(batch_of_logs)} logs, DB not connected. Logs might be lost if queue fills.")
            # Consider re-queueing or saving to a temporary file if DB is critical
            return

        logger.debug(f"Writing batch of {len(batch_of_logs)} logs to DB.")
        # Group logs by type (table name) for efficient batch insertion
        # This example assumes each log entry is a tuple: (table_name, query, params)
        # or a more structured object indicating which _execute_batch_insert method to call.
        # For simplicity, this example processes them one by one, but batch insert is better.

        # A more optimized approach would use executemany for each table.
        # Example:
        # grouped_logs = {} # {'table_name': [(params_tuple_1), (params_tuple_2), ...]}
        # for log_type, query, params in batch_of_logs:
        #     grouped_logs.setdefault(log_type, []).append(params)
        # for log_type, params_list in grouped_logs.items():
        #     self._execute_batch_insert(log_type, params_list)

        # Simplified sequential execution for this example (less performant than true batch SQL)
        with self._db_access_lock: # Protect DB access
            if self.conn is None: return # Double check after acquiring lock

            cursor = None
            try:
                with self.conn: # Use transaction context manager
                    cursor = self.conn.cursor()
                    for log_item in batch_of_logs:
                        query = log_item["query"]
                        params = log_item["params"]
                        try:
                            cursor.execute(query, params)
                        except Exception as exec_e:
                            logger.error(f"Error executing single query in batch: {exec_e}. Query: {query}, Params: {params}")
                            # Optionally, try to commit successful ones if partial success is okay,
                            # or rollback the entire batch on any error.
                            # For now, continue processing other logs in the batch but log the error.
                # Commit is handled by `with self.conn:` if successful, rollback if exception
                logger.debug(f"Successfully wrote batch of {len(batch_of_logs)} logs (individual execution).")
            except Exception as batch_e:
                logger.error(f"Error during batch DB write: {batch_e}", exc_info=True)
                # Handle potential rollback here if not using `with self.conn`
            finally:
                if cursor:
                    cursor.close()


    def _enqueue_log(self, query_type, params_dict):
        """Helper to put log data onto the internal queue."""
        # This method constructs the actual SQL query and parameters based on query_type and params_dict
        # and then puts {"query": sql_query, "params": sql_params_tuple} into the queue.
        # This is highly dependent on your table schemas and logging methods.

        # For simplicity, let's assume logging methods will construct the query and params
        # and directly call _put_on_queue.
        pass # This method might not be needed if logging methods call _put_on_queue directly.

    def _put_on_queue(self, sql_query, sql_params_tuple):
        """Puts a formatted SQL query and its parameters onto the log queue."""
        if self._stop_writer_event.is_set(): # Don't enqueue if shutting down
            logger.warning(f"Attempted to log while writer is stopping. Query: {sql_query[:50]}...")
            return

        try:
            log_item = {"query": sql_query, "params": sql_params_tuple}
            self._log_queue.put(log_item, timeout=1.0) # Put with a timeout to avoid blocking indefinitely
        except queue.Full:
            logger.error("Log queue is full! Some logs might be lost. Consider increasing queue size or DB write performance.")
            # Implement strategies for full queue: e.g., drop oldest, save to temp file.

    def _get_current_timestamp_for_db(self):
        """Returns current timestamp appropriate for the DB type."""
        # PostgreSQL TIMESTAMPTZ expects timezone-aware datetime objects or ISO strings.
        # SQLite TEXT can store ISO strings.
        return datetime.datetime.now(datetime.timezone.utc) # Store as UTC


    # --- Public Logging Methods (Now enqueue logs) ---
    # Each method now constructs the SQL and params, then puts them on the queue.

    def log_process_event(self, event_type, job_id=None, robot_id=None, details=None):
        ts = self._get_current_timestamp_for_db()
        details_json = json.dumps(details,ensure_ascii=False) if details is not None else None # ensure_ascii for non-English chars
        query = """
        INSERT INTO process_events (timestamp, job_id, robot_id, event_type, details)
        VALUES (%s, %s, %s, %s, %s);
        """ # Use %s for psycopg2, ? for sqlite3
        params = (ts, job_id, robot_id, event_type, details_json)
        self._put_on_queue(query, params)
        # SQLite would use: "VALUES (?, ?, ?, ?, ?);"

    def log_sensor_reading(self, sensor_name, sensor_value, robot_id=None, job_id=None, extra_info=None):
        ts = self._get_current_timestamp_for_db()
        extra_info_json = json.dumps(extra_info, ensure_ascii=False) if extra_info is not None else None
        query = """
        INSERT INTO sensor_readings (timestamp, robot_id, sensor_name, sensor_value, sensor_extra_info, job_id)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        params = (ts, robot_id, sensor_name, sensor_value, extra_info_json, job_id)
        self._put_on_queue(query, params)

    def log_ai_prediction(self, prediction_result, input_data=None, robot_id=None, job_id=None):
        if not prediction_result: return
        ts = self._get_current_timestamp_for_db()
        input_json = json.dumps(input_data.tolist() if hasattr(input_data, 'tolist') else input_data, ensure_ascii=False) if input_data is not None else None
        probs_json = json.dumps(prediction_result.get('probabilities'), ensure_ascii=False) if prediction_result.get('probabilities') is not None else None
        raw_output_json = json.dumps(prediction_result.get('raw_output'), ensure_ascii=False) if prediction_result.get('raw_output') is not None else None
        query = """
        INSERT INTO ai_predictions (timestamp, job_id, robot_id, input_data,
                                    predicted_class, probabilities, confidence, predicted_score, raw_output)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (ts, job_id, robot_id, input_json,
                  prediction_result.get('predicted_class'), probs_json,
                  prediction_result.get('confidence'), prediction_result.get('predicted_score'),
                  raw_output_json)
        self._put_on_queue(query, params)

    def log_quality_evaluation(self, evaluation_result, ai_prediction_id=None, sensor_summary=None, robot_id=None, job_id=None):
        if not evaluation_result: return
        ts = self._get_current_timestamp_for_db()
        sensor_summary_json = json.dumps(sensor_summary, ensure_ascii=False) if sensor_summary is not None else None
        issues_json = json.dumps(evaluation_result.get('detected_issues'), ensure_ascii=False) if evaluation_result.get('detected_issues') is not None else None
        query = """
        INSERT INTO quality_evaluations (timestamp, job_id, robot_id, ai_prediction_id,
                                         sensor_data_summary, combined_status, detected_issues, numeric_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        params = (ts, job_id, robot_id, ai_prediction_id, sensor_summary_json,
                  evaluation_result.get('combined_status'), issues_json,
                  evaluation_result.get('numeric_score'))
        self._put_on_queue(query, params)

    def log_adaptive_control_action(self, adjustments, triggering_eval_id=None, status='GENERATED', actual_params=None, robot_id=None, job_id=None):
        # 'status' can be 'GENERATED', 'APPLIED', 'REJECTED', 'FAILED_TO_APPLY', 'ERROR_DURING_APPLY'
        # 'actual_params' can log the parameters that were actually sent to the robot after limits/rounding
        if not adjustments and status == 'NO_ADJUSTMENT_NEEDED': # Log if explicitly no adjustment
             adjustments_json = None
        elif not adjustments:
             return # Don't log if adjustments are None and status isn't explicitly 'NO_ADJUSTMENT_NEEDED'
        else:
             adjustments_json = json.dumps(adjustments, ensure_ascii=False)

        ts = self._get_current_timestamp_for_db()
        actual_params_json = json.dumps(actual_params, ensure_ascii=False) if actual_params is not None else None
        query = """
        INSERT INTO adaptive_control_actions (timestamp, job_id, robot_id,
                                              triggering_evaluation_id, suggested_adjustments, action_taken_status, actual_applied_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        params = (ts, job_id, robot_id, triggering_eval_id, adjustments_json, status, actual_params_json)
        self._put_on_queue(query, params)

    def log_robot_status(self, robot_id, status_data, job_id=None):
        if not status_data: return
        ts = self._get_current_timestamp_for_db()
        status_json = json.dumps(status_data, ensure_ascii=False)
        query = """
        INSERT INTO robot_status_logs (timestamp, robot_id, status_data, job_id)
        VALUES (%s, %s, %s, %s);
        """
        params = (ts, robot_id, status_json, job_id)
        self._put_on_queue(query, params)


    # --- Data Retrieval Methods (Example) ---
    def get_recent_process_events(self, job_id=None, limit=10):
        """Retrieves recent process events, optionally filtered by job_id."""
        if not self._ensure_connection(): return []
        query = "SELECT * FROM process_events"
        params = []
        if job_id:
            query += " WHERE job_id = %s" # or ? for sqlite
            params.append(job_id)
        query += " ORDER BY timestamp DESC LIMIT %s;" # or ? for sqlite
        params.append(limit)

        with self._db_access_lock:
            if self.conn is None: return []
            cursor = None
            try:
                # For PostgreSQL, use RealDictCursor for dictionary-like rows
                cursor_factory = psycopg2.extras.RealDictCursor if self.db_type == 'postgresql' else None
                cursor = self.conn.cursor(cursor_factory=cursor_factory)
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                # Convert RealDictRow to dict for psycopg2, sqlite3.Row already acts like dict
                return [dict(row) for row in rows] if self.db_type == 'postgresql' else rows
            except Exception as e:
                logger.error(f"Error fetching recent process events: {e}", exc_info=True)
                return []
            finally:
                if cursor:
                    cursor.close()


    def close_connection(self):
        """Stops the batch writer thread and closes the database connection."""
        logger.info("DataLoggerDB shutdown initiated.")
        # Signal and wait for the batch writer thread to stop
        if self._batch_writer_thread and self._batch_writer_thread.is_alive():
            logger.info("Stopping database batch writer thread...")
            self._stop_writer_event.set()
            self._batch_writer_thread.join(timeout=self.BATCH_TIMEOUT_SEC + 2.0) # Give extra time
            if self._batch_writer_thread.is_alive():
                 logger.warning("DBBatchWriter thread did not terminate gracefully.")
            else:
                 logger.info("DBBatchWriter thread stopped.")
        self._batch_writer_thread = None


        with self._db_access_lock:
            if self.conn:
                try:
                    self.conn.close()
                    logger.info("Database connection closed.")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
                finally:
                    self.conn = None # Ensure connection is marked as None
                    self._initialized = False # Mark as uninitialized for potential re-init


# Example Usage (Typically, other modules would import and use this singleton)
if __name__ == '__main__':
    # Configure logging for direct script run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    logger.info("--- Data Logger DB (Advanced) Example ---")

    # Override config for testing if needed (config.py might use environment variables)
    # For this example, assume config.py uses defaults that work for SQLite or a test PG DB
    if config.DB_TYPE == 'postgresql' and (not config.DB_USER or config.DB_PASSWORD == 'your_secure_password'):
        logger.warning("PostgreSQL selected but credentials might be defaults. Ensure test DB is configured.")

    db_logger_instance = None
    try:
        # Get the singleton instance (this also initializes it)
        db_logger_instance = DataLoggerDB()

        # --- Example Logging Operations ---
        if db_logger_instance.conn: # Check if connection was successful
            job_id_example = "ADV_JOB_001"
            robot_id_example = 1

            db_logger_instance.log_process_event("SYSTEM_STARTUP", job_id=None, details={"message": "System started successfully"})
            db_logger_instance.log_sensor_reading("temperature_c", 35.5, robot_id=robot_id_example, job_id=job_id_example, extra_info={"location": "ambient"})
            dummy_ai_pred = {'predicted_class': 1, 'probabilities': [0.1, 0.8, 0.05, 0.05], 'confidence': 0.8, 'predicted_score': None}
            db_logger_instance.log_ai_prediction(dummy_ai_pred, input_data=[1.0, 2.1, 3.2], robot_id=robot_id_example, job_id=job_id_example)

            logger.info("Dummy logs enqueued. Waiting for batch writer to process...")
            time.sleep(config.DB_LOG_BATCH_TIMEOUT_SEC + 1) # Wait for batch timeout to ensure write

            logger.info("\n--- Fetching recent process events (example) ---")
            recent_events = db_logger_instance.get_recent_process_events(limit=5)
            if recent_events:
                for row in recent_events:
                    # For SQLite, row is sqlite3.Row. For PG with RealDictCursor, row is dict.
                    if isinstance(row, sqlite3.Row): # sqlite3.Row can be accessed by index or key
                        logger.info(f"Event ID: {row['id']}, Type: {row['event_type']}, Job: {row['job_id']}, Details: {row['details']}")
                    else: # Assuming dict for PG
                         logger.info(f"Event: {row}")

            else:
                logger.info("No recent events found or DB query failed.")
        else:
            logger.error("DB Logger example could not connect to database. Logging calls were queued but not written.")

    except Exception as e:
        logger.critical(f"Error in DataLoggerDB example: {e}", exc_info=True)
    finally:
        if db_logger_instance:
            db_logger_instance.close_connection() # Ensure cleanup

        # Clean up dummy SQLite DB file if created by this example
        if config.DB_TYPE == 'sqlite' and os.path.exists(config.SQLITE_DB_PATH) and "test_welding_log_advanced.db" in config.SQLITE_DB_PATH:
            logger.info(f"Removing test SQLite database: {config.SQLITE_DB_PATH}")
            try:
                os.remove(config.SQLITE_DB_PATH)
            except Exception as e_remove:
                logger.error(f"Failed to remove test database: {e_remove}")


    logger.info("--- Data Logger DB (Advanced) Example Finished ---")