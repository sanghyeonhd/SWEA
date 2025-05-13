# data_logger_db.py
# Description: Handles logging of all process data, sensor readings,
#              AI predictions, quality evaluations, and robot status to a database.

import sqlite3
import json
import datetime
import logging
import threading # For thread-safe database access

from src import config # For DB_PATH or other DB connection settings

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Schema (Example for SQLite) ---
# Table: process_events
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL (ISO 8601 format)
#   job_id TEXT
#   event_type TEXT NOT NULL (e.g., 'STATE_CHANGE', 'ERROR', 'PARAMETER_UPDATE', 'WELD_START', 'WELD_END')
#   details TEXT (JSON string for event-specific data)

# Table: sensor_readings
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL
#   robot_id INTEGER
#   sensor_name TEXT NOT NULL
#   sensor_value REAL or TEXT (depending on sensor type)
#   job_id TEXT (Optional, if sensor reading is tied to a specific job)
#   # For image/point cloud data, store path to file or use blob (not ideal for large files in SQLite)

# Table: ai_predictions
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL
#   job_id TEXT
#   robot_id INTEGER
#   input_data TEXT (JSON string of sensor values used for prediction)
#   predicted_class INTEGER (For classification)
#   probabilities TEXT (JSON array of probabilities for classification)
#   confidence REAL
#   predicted_score REAL (For regression)
#   raw_output TEXT (JSON string of model's raw output)

# Table: quality_evaluations
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL
#   job_id TEXT
#   robot_id INTEGER
#   ai_prediction_id INTEGER (FK to ai_predictions table)
#   sensor_data_summary TEXT (JSON string of relevant sensor data at eval time)
#   combined_status TEXT (e.g., 'Good', 'Warning', 'Critical')
#   detected_issues TEXT (JSON array of detected issues)
#   numeric_score REAL (Overall quality score if applicable)

# Table: adaptive_control_actions
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL
#   job_id TEXT
#   robot_id INTEGER
#   triggering_evaluation_id INTEGER (FK to quality_evaluations table)
#   suggested_adjustments TEXT (JSON string of parameter adjustments)
#   action_taken_status TEXT (e.g., 'APPLIED', 'REJECTED', 'FAILED_TO_APPLY')

# Table: robot_status_logs
#   id INTEGER PRIMARY KEY AUTOINCREMENT
#   timestamp TEXT NOT NULL
#   robot_id INTEGER NOT NULL
#   status_data TEXT (JSON string of robot status, e.g., position, speed, errors)
#   job_id TEXT


class DataLoggerDB:
    """
    Manages database connections and provides methods to log various system data.
    Uses SQLite for this example.
    """
    _instance = None
    _lock = threading.Lock() # For thread-safe singleton instantiation and DB access

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to ensure only one DB connection manager instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # Initialize here if not passing db_path in __init__ every time
                # cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path=None):
        # Allow re-initialization with a new path for testing, but typically called once
        with self._lock:
            # Check if already initialized by this instance.
            # A more robust singleton might handle this differently.
            if hasattr(self, '_initialized') and self._initialized and self.db_path == db_path:
                return

            if db_path is None:
                self.db_path = getattr(config, 'DATABASE_PATH', 'welding_data.db')
            else:
                self.db_path = db_path

            self.conn = None
            self._ensure_connection() # Establish connection
            self._create_tables()     # Create tables if they don't exist
            self._initialized = True
            logger.info(f"DataLoggerDB initialized with database: {self.db_path}")

    def _ensure_connection(self):
        """Ensures a database connection is active."""
        if self.conn is None:
            try:
                # `check_same_thread=False` is needed if multiple threads will use this
                # DataLoggerDB instance (which is a singleton).
                # For robust multi-threading, consider a connection pool or a dedicated writer thread.
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row # Access columns by name
                logger.info(f"Successfully connected to SQLite database: {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to SQLite database {self.db_path}: {e}")
                self.conn = None # Ensure conn is None if connection fails

    def _execute_query(self, query, params=(), fetch_one=False, fetch_all=False, commit=False):
        """Helper method to execute SQL queries safely."""
        if self.conn is None:
            logger.error("Database not connected. Cannot execute query.")
            return None

        cursor = None
        try:
            with self._lock: # Ensure thread-safe execution of individual queries
                cursor = self.conn.cursor()
                cursor.execute(query, params)
                if commit:
                    self.conn.commit()
                    return cursor.lastrowid # Return last inserted row ID for INSERTs

                if fetch_one:
                    return cursor.fetchone()
                if fetch_all:
                    return cursor.fetchall()
                return True # For non-SELECT queries without commit (e.g., table creation)
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}\nQuery: {query}\nParams: {params}")
            # Consider re-establishing connection on certain errors
            return None
        finally:
            if cursor:
                cursor.close()

    def _create_tables(self):
        """Creates database tables if they do not already exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS process_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                job_id TEXT,
                event_type TEXT NOT NULL,
                details TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                robot_id INTEGER,
                sensor_name TEXT NOT NULL,
                sensor_value REAL,
                sensor_extra_info TEXT, -- For JSON string of more complex data like image path
                job_id TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ai_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                job_id TEXT,
                robot_id INTEGER,
                input_data TEXT,
                predicted_class INTEGER,
                probabilities TEXT,
                confidence REAL,
                predicted_score REAL,
                raw_output TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS quality_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                job_id TEXT,
                robot_id INTEGER,
                ai_prediction_id INTEGER,
                sensor_data_summary TEXT,
                combined_status TEXT,
                detected_issues TEXT,
                numeric_score REAL,
                FOREIGN KEY (ai_prediction_id) REFERENCES ai_predictions (id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS adaptive_control_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                job_id TEXT,
                robot_id INTEGER,
                triggering_evaluation_id INTEGER,
                suggested_adjustments TEXT,
                action_taken_status TEXT,
                FOREIGN KEY (triggering_evaluation_id) REFERENCES quality_evaluations (id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS robot_status_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                robot_id INTEGER NOT NULL,
                status_data TEXT,
                job_id TEXT
            );
            """
        ]
        for query in queries:
            if not self._execute_query(query):
                 logger.error(f"Failed to create table with query: {query[:50]}...") # Log start of query
                 # This indicates a serious problem if tables can't be created.
                 # Potentially raise an exception or set a flag to disable logging.

        logger.info("Database tables checked/created successfully.")

    def _get_current_timestamp(self):
        """Returns current timestamp in ISO 8601 format."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # --- Logging Methods ---

    def log_process_event(self, event_type, job_id=None, details=None):
        """Logs a general process event."""
        query = """
        INSERT INTO process_events (timestamp, job_id, event_type, details)
        VALUES (?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        details_json = json.dumps(details) if details is not None else None
        return self._execute_query(query, (ts, job_id, event_type, details_json), commit=True)

    def log_sensor_reading(self, sensor_name, sensor_value, robot_id=None, job_id=None, extra_info=None):
        """Logs a single sensor reading."""
        query = """
        INSERT INTO sensor_readings (timestamp, robot_id, sensor_name, sensor_value, sensor_extra_info, job_id)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        extra_info_json = json.dumps(extra_info) if extra_info is not None else None
        return self._execute_query(query, (ts, robot_id, sensor_name, sensor_value, extra_info_json, job_id), commit=True)

    def log_ai_prediction(self, prediction_result, input_data=None, robot_id=None, job_id=None):
        """Logs an AI prediction result."""
        if not prediction_result: return None
        query = """
        INSERT INTO ai_predictions (timestamp, job_id, robot_id, input_data,
                                    predicted_class, probabilities, confidence, predicted_score, raw_output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        input_json = json.dumps(input_data.tolist() if hasattr(input_data, 'tolist') else input_data) if input_data is not None else None
        probs_json = json.dumps(prediction_result.get('probabilities')) if prediction_result.get('probabilities') is not None else None
        # Assuming 'raw_output' might be the direct model output before softmax if needed
        raw_output_json = json.dumps(prediction_result.get('raw_output')) if prediction_result.get('raw_output') is not None else None

        return self._execute_query(query, (
            ts, job_id, robot_id, input_json,
            prediction_result.get('predicted_class'),
            probs_json,
            prediction_result.get('confidence'),
            prediction_result.get('predicted_score'),
            raw_output_json
        ), commit=True)

    def log_quality_evaluation(self, evaluation_result, ai_prediction_id=None, sensor_summary=None, robot_id=None, job_id=None):
        """Logs a quality evaluation result."""
        if not evaluation_result: return None
        query = """
        INSERT INTO quality_evaluations (timestamp, job_id, robot_id, ai_prediction_id,
                                         sensor_data_summary, combined_status, detected_issues, numeric_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        sensor_summary_json = json.dumps(sensor_summary) if sensor_summary is not None else None
        issues_json = json.dumps(evaluation_result.get('detected_issues')) if evaluation_result.get('detected_issues') is not None else None

        return self._execute_query(query, (
            ts, job_id, robot_id, ai_prediction_id,
            sensor_summary_json,
            evaluation_result.get('combined_status'),
            issues_json,
            evaluation_result.get('numeric_score') # Assuming it has 'numeric_score' from evaluator
        ), commit=True)

    def log_adaptive_control_action(self, adjustments, triggering_eval_id=None, status='APPLIED', robot_id=None, job_id=None):
        """Logs an adaptive control action."""
        if not adjustments: return None
        query = """
        INSERT INTO adaptive_control_actions (timestamp, job_id, robot_id,
                                              triggering_evaluation_id, suggested_adjustments, action_taken_status)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        adjustments_json = json.dumps(adjustments)

        return self._execute_query(query, (
            ts, job_id, robot_id,
            triggering_eval_id,
            adjustments_json,
            status
        ), commit=True)

    def log_robot_status(self, robot_id, status_data, job_id=None):
        """Logs robot status data."""
        if not status_data: return None
        query = """
        INSERT INTO robot_status_logs (timestamp, robot_id, status_data, job_id)
        VALUES (?, ?, ?, ?);
        """
        ts = self._get_current_timestamp()
        status_json = json.dumps(status_data)

        return self._execute_query(query, (ts, robot_id, status_json, job_id), commit=True)


    def close_connection(self):
        """Closes the database connection."""
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None
                self._initialized = False # Mark as uninitialized for potential re-init
                logger.info("Database connection closed.")


# Example Usage (Typically, other modules would import and use this)
if __name__ == '__main__':
    logger.info("--- Data Logger DB Example ---")

    # Use a temporary in-memory DB or a test file for example
    # To use a file, ensure config.DATABASE_PATH is set or pass db_path
    class DummyConfig:
        DATABASE_PATH = 'test_welding_log.db' # Example file
    config = DummyConfig()

    # Get the singleton instance
    db_logger = DataLoggerDB(db_path=config.DATABASE_PATH)
    db_logger_same_instance = DataLoggerDB() # Should be the same instance

    logger.info(f"Is db_logger same as db_logger_same_instance? {db_logger is db_logger_same_instance}")


    # --- Example Logging Operations ---
    job_id_example = "JOB_XYZ_123"
    robot_id_example = 1

    # Log Process Event
    event_id = db_logger.log_process_event("STATE_CHANGE", job_id=job_id_example, details={"old_state": "IDLE", "new_state": "LOADING"})
    logger.info(f"Logged process event with ID: {event_id}")

    # Log Sensor Reading
    sensor_id = db_logger.log_sensor_reading("temperature", 350.75, robot_id=robot_id_example, job_id=job_id_example, extra_info={"unit": "C"})
    logger.info(f"Logged sensor reading with ID: {sensor_id}")
    db_logger.log_sensor_reading("arc_current", 155.2, robot_id=robot_id_example, job_id=job_id_example)


    # Log AI Prediction
    dummy_ai_pred = {'predicted_class': 0, 'probabilities': [0.9, 0.05, 0.03, 0.02], 'confidence': 0.9}
    dummy_input_data = [150.0, 22.0, 300.0, 320.0]
    ai_pred_id = db_logger.log_ai_prediction(dummy_ai_pred, input_data=dummy_input_data, robot_id=robot_id_example, job_id=job_id_example)
    logger.info(f"Logged AI prediction with ID: {ai_pred_id}")

    # Log Quality Evaluation
    dummy_quality_eval = {'combined_status': 'Good', 'detected_issues': [], 'numeric_score': 0.95}
    dummy_sensor_summary = {'avg_temp': 340.0, 'arc_stability': 0.88}
    quality_eval_id = db_logger.log_quality_evaluation(dummy_quality_eval, ai_prediction_id=ai_pred_id, sensor_summary=dummy_sensor_summary, robot_id=robot_id_example, job_id=job_id_example)
    logger.info(f"Logged quality evaluation with ID: {quality_eval_id}")

    # Log Adaptive Control Action
    dummy_adjustments = {'speed': -10, 'current': +5}
    control_action_id = db_logger.log_adaptive_control_action(dummy_adjustments, triggering_eval_id=quality_eval_id, robot_id=robot_id_example, job_id=job_id_example)
    logger.info(f"Logged adaptive control action with ID: {control_action_id}")

    # Log Robot Status
    dummy_robot_status = {"position_tcp": [100.1, 200.2, 300.3], "speed_override": 100, "errors": []}
    robot_status_id = db_logger.log_robot_status(robot_id_example, dummy_robot_status, job_id=job_id_example)
    logger.info(f"Logged robot status with ID: {robot_status_id}")

    # --- Example Query (Illustrative) ---
    logger.info("\n--- Fetching recent process events ---")
    recent_events = db_logger._execute_query("SELECT * FROM process_events ORDER BY timestamp DESC LIMIT 5;", fetch_all=True)
    if recent_events:
        for row in recent_events:
            logger.info(f"Event ID: {row['id']}, Type: {row['event_type']}, Job: {row['job_id']}, Details: {row['details']}")

    # Close connection when application exits (or manage centrally)
    db_logger.close_connection()

    # Remove the test database file after example run
    import os
    if os.path.exists(config.DATABASE_PATH):
        logger.info(f"Removing test database: {config.DATABASE_PATH}")
        os.remove(config.DATABASE_PATH)

    logger.info("--- Data Logger DB Example Finished ---")