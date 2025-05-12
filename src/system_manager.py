# system_manager.py
# Description: Manages the lifecycle and coordination of all system modules.
#              Initializes, starts, stops, and monitors core components.

import time
import logging
import threading # For managing module threads or background tasks
import signal # For handling system shutdown signals

import config # Main configuration for the entire system
# Import all core modules that this manager will control
from robot_control_interface import RobotControlInterface
from sensor_data_handler import SensorDataHandler # Placeholder - Needs actual implementation
from ai_inference_engine import AIInferenceEngine
from quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl
from welding_process_manager import WeldingProcessManager
from data_logger_db import DataLoggerDB
# from hmi_interface import HMIInterface # Placeholder for HMI communication

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemManager:
    """
    Manages the overall digital twin system, including initialization,
    lifecycle of core modules, and orchestration.
    """
    def __init__(self):
        logger.info("Initializing System Manager...")

        # --- Configuration Loading ---
        # `config` module is imported directly. Specific checks or loading logic can go here.
        # Example: Validate essential config parameters
        if not hasattr(config, 'ROBOT_CONFIGS'):
            logger.warning("ROBOT_CONFIGS not found in config. System may not function correctly with robots.")
        if not hasattr(config, 'DATABASE_PATH'):
             logger.warning("DATABASE_PATH not found in config. Data logging might use default or fail.")


        # --- Module Initialization ---
        # Order of initialization might matter based on dependencies.
        self.data_logger = None
        self.robot_interface = None
        self.sensor_handler = None
        self.ai_engine = None
        self.quality_controller = None
        self.process_manager = None
        # self.hmi_interface = None # For HMI communication

        self._initialize_modules()

        self.is_running = False
        self._shutdown_event = threading.Event() # Event to signal shutdown

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler) # Handle `kill` command

    def _initialize_modules(self):
        """Initializes all core system modules."""
        logger.info("Initializing core system modules...")
        try:
            # 1. Data Logger (often needed by other modules for logging during init)
            self.data_logger = DataLoggerDB() # Uses singleton, path from config
            logger.info("Data Logger DB initialized.")

            # 2. Robot Control Interface
            self.robot_interface = RobotControlInterface() # Reads ROBOT_CONFIGS from config
            logger.info("Robot Control Interface initialized.")

            # 3. Sensor Data Handler (Placeholder - needs actual implementation)
            # This might start threads for polling sensors or connect to data streams.
            self.sensor_handler = SensorDataHandler(self.data_logger) # Pass logger if needed
            logger.info("Sensor Data Handler initialized (Placeholder).")

            # 4. AI Inference Engine
            self.ai_engine = AIInferenceEngine() # Loads model and scaler from config paths
            logger.info("AI Inference Engine initialized.")

            # 5. Quality Evaluator and Adaptive Control
            self.quality_controller = QualityEvaluatorAdaptiveControl() # Uses config for rules/thresholds
            logger.info("Quality Evaluator & Adaptive Control initialized.")

            # 6. Welding Process Manager (depends on many of the above)
            self.process_manager = WeldingProcessManager(
                robot_interface=self.robot_interface,
                ai_engine=self.ai_engine,
                quality_controller=self.quality_controller,
                sensor_handler=self.sensor_handler, # Pass the instance
                # data_logger=self.data_logger # Pass if WPM logs directly, or WPM passes data to logger
            )
            logger.info("Welding Process Manager initialized.")

            # 7. HMI Interface (Placeholder)
            # self.hmi_interface = HMIInterface(self.process_manager, self.data_logger)
            # logger.info("HMI Interface initialized (Placeholder).")

            logger.info("All core modules initialized successfully.")
            self.log_system_event("SYSTEM_INITIALIZED", "All modules initialized.")

        except Exception as e:
            logger.critical(f"Fatal error during module initialization: {e}", exc_info=True)
            # Potentially try to clean up already initialized modules before exiting
            self.shutdown(graceful=False) # Attempt to shutdown what was started
            raise SystemExit(f"System initialization failed: {e}")


    def log_system_event(self, event_type, message, details=None):
        """Helper to log system-level events via DataLoggerDB."""
        if self.data_logger:
            log_details = {"message": message}
            if details:
                log_details.update(details)
            self.data_logger.log_process_event(event_type="SYSTEM_" + event_type, details=log_details)
        else:
            logger.warning(f"DataLogger not available. System Event not logged: {event_type} - {message}")

    def start(self):
        """Starts all managed services and the main system loop."""
        if self.is_running:
            logger.warning("System is already running.")
            return

        logger.info("Starting System Manager and core services...")
        self.is_running = True
        self._shutdown_event.clear()
        self.log_system_event("SYSTEM_START", "System Manager started.")

        # --- Start Core Services/Threads ---
        # 1. Connect to Robots
        if self.robot_interface:
            logger.info("Connecting to robots...")
            # connect_all might be blocking, or could return future/thread objects
            # For this example, assume it's blocking for simplicity during startup
            connected_robots = self.robot_interface.connect_all()
            if connected_robots:
                 logger.info(f"Successfully connected to robots: {connected_robots}")
                 self.log_system_event("ROBOT_CONNECTION", "Robots connected.", {"connected_ids": connected_robots})
            else:
                 logger.warning("Failed to connect to any robots. System may operate in a limited mode.")
                 self.log_system_event("ROBOT_CONNECTION_FAILED", "No robots connected.")


        # 2. Start Sensor Data Handler (if it has a start method, e.g., for polling threads)
        if self.sensor_handler and hasattr(self.sensor_handler, 'start_collection'):
            logger.info("Starting sensor data collection...")
            try:
                self.sensor_handler.start_collection() # This is a hypothetical method
                self.log_system_event("SENSOR_HANDLER_START", "Sensor data collection started.")
            except Exception as e:
                logger.error(f"Failed to start sensor data collection: {e}", exc_info=True)
                self.log_system_event("SENSOR_HANDLER_ERROR", f"Failed to start: {e}")


        # 3. Start HMI Interface (if it runs its own server/loop)
        # if self.hmi_interface and hasattr(self.hmi_interface, 'start'):
        #     logger.info("Starting HMI interface...")
        #     self.hmi_interface.start() # Hypothetical

        logger.info("System Manager is now running. Waiting for external triggers or jobs.")
        # Main loop for System Manager (could monitor module health, handle high-level commands)
        try:
            while not self._shutdown_event.is_set():
                # --- System Health Monitoring (Example) ---
                # if self.robot_interface and not self.robot_interface.are_all_essential_robots_connected():
                #     logger.warning("Essential robot disconnected! Taking action...")
                #     self.log_system_event("HEALTH_CHECK_WARNING", "Essential robot disconnected.")
                #     # Potentially pause Process Manager or notify HMI

                # --- Handle External Commands (Placeholder) ---
                # This could be from a message queue, API endpoint, or HMI
                # command = self.check_for_external_commands()
                # if command:
                #    self.process_external_command(command)

                # For now, just sleep and wait for shutdown signal or job start via Process Manager
                time.sleep(1) # Main loop heartbeat

        except KeyboardInterrupt: # Should be caught by signal_handler, but as a fallback
            logger.info("KeyboardInterrupt caught in main loop. Initiating shutdown...")
            self.shutdown()
        except Exception as e:
            logger.critical(f"Unhandled exception in System Manager main loop: {e}", exc_info=True)
            self.log_system_event("SYSTEM_CRITICAL_ERROR", f"Unhandled exception: {e}")
            self.shutdown(graceful=False) # Non-graceful shutdown on critical error
        finally:
            if self.is_running: # If loop exited without shutdown being called by signal
                self.shutdown()


    def shutdown(self, graceful=True):
        """Shuts down all managed services and modules."""
        if not self.is_running and not self._shutdown_event.is_set(): # Prevent multiple shutdown calls
            logger.info("System is not running or already shutting down.")
            # return # Or allow re-entry for cleanup if called multiple times

        logger.info(f"Initiating {'graceful' if graceful else 'forced'} system shutdown...")
        self.is_running = False
        self._shutdown_event.set() # Signal all loops/threads to stop

        self.log_system_event("SYSTEM_SHUTDOWN_START", f"Graceful: {graceful}")

        # --- Stop Core Services/Threads (in reverse order of start or based on dependencies) ---

        # 1. Stop Welding Process Manager (tell it to stop any active jobs)
        if self.process_manager:
            logger.info("Stopping Welding Process Manager...")
            try:
                self.process_manager.stop_current_job() # Request current job to stop
                # Wait for its thread to finish if it's robustly designed
                if self.process_manager.process_thread and self.process_manager.process_thread.is_alive():
                     logger.info("Waiting for Process Manager thread to complete...")
                     self.process_manager.process_thread.join(timeout=10) # Give it time
                     if self.process_manager.process_thread.is_alive():
                          logger.warning("Process Manager thread did not terminate gracefully.")
            except Exception as e:
                logger.error(f"Error stopping Welding Process Manager: {e}", exc_info=True)

        # 2. Stop Sensor Data Handler
        if self.sensor_handler and hasattr(self.sensor_handler, 'stop_collection'):
            logger.info("Stopping sensor data collection...")
            try:
                self.sensor_handler.stop_collection() # Hypothetical
                self.log_system_event("SENSOR_HANDLER_STOP", "Sensor data collection stopped.")
            except Exception as e:
                 logger.error(f"Error stopping sensor data collection: {e}", exc_info=True)

        # 3. Disconnect from Robots
        if self.robot_interface:
            logger.info("Disconnecting from robots...")
            try:
                self.robot_interface.disconnect_all()
                self.log_system_event("ROBOT_DISCONNECTION", "Robots disconnected.")
            except Exception as e:
                 logger.error(f"Error disconnecting robots: {e}", exc_info=True)


        # 4. Stop HMI Interface
        # if self.hmi_interface and hasattr(self.hmi_interface, 'stop'):
        #     logger.info("Stopping HMI interface...")
        #     self.hmi_interface.stop()

        # 5. Close Data Logger connection (should be one of the last)
        if self.data_logger:
            logger.info("Closing Data Logger connection...")
            try:
                self.data_logger.close_connection()
                # No event logging after this point via data_logger
            except Exception as e:
                 logger.error(f"Error closing data logger: {e}", exc_info=True)


        logger.info("System Manager shutdown complete.")
        # After this, the main script can exit.

    def _signal_handler(self, signum, frame):
        """Handles OS signals like SIGINT (Ctrl+C) and SIGTERM."""
        logger.warning(f"Received signal {signal.Signals(signum).name}. Initiating shutdown...")
        if not self._shutdown_event.is_set(): # Prevent re-entry if already shutting down
             self.shutdown()
        else:
             logger.info("Shutdown already in progress.")

    # --- Methods for external control (e.g., from HMI or API) ---
    def start_job_externally(self, job_id):
        """Allows an external system to request a job start."""
        if not self.is_running:
            logger.error(f"Cannot start job '{job_id}': System Manager is not running.")
            return False
        if self.process_manager:
            logger.info(f"External request to start job: {job_id}")
            success = self.process_manager.start_welding_job(job_id)
            self.log_system_event("EXTERNAL_JOB_START_REQUEST", f"Job: {job_id}, Success: {success}")
            return success
        else:
            logger.error("Process Manager not available to start job.")
            return False

    def stop_job_externally(self):
        """Allows an external system to request stopping the current job."""
        if not self.is_running:
            logger.error("Cannot stop job: System Manager is not running.")
            return False
        if self.process_manager:
            logger.info("External request to stop current job.")
            success = self.process_manager.stop_current_job()
            self.log_system_event("EXTERNAL_JOB_STOP_REQUEST", f"Success: {success}")
            return success
        else:
            logger.error("Process Manager not available to stop job.")
            return False

    def get_system_status_externally(self):
        """Provides a summary of the system status for external queries."""
        if not self.is_running and not self._shutdown_event.is_set() and not self.data_logger: # If not even initialized
             return {"manager_status": "NOT_INITIALIZED", "error": "System modules not loaded."}

        status = {
            "manager_status": "RUNNING" if self.is_running else ("SHUTTING_DOWN" if self._shutdown_event.is_set() else "STOPPED"),
            "process_manager_status": self.process_manager.get_manager_status() if self.process_manager else None,
            "robot_interface_status": {
                "connected_robots": self.robot_interface.connected_robot_ids if self.robot_interface else []
            },
            "ai_engine_status": {
                "model_loaded": self.ai_engine.model_loaded if self.ai_engine else False,
                "scaler_loaded": self.ai_engine.scaler_loaded if self.ai_engine else False
            },
            # Add status from other modules (sensor_handler, quality_controller, data_logger)
            "sensor_handler_status": self.sensor_handler.get_status() if self.sensor_handler and hasattr(self.sensor_handler, 'get_status') else "UNKNOWN", # Hypothetical
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        return status

# --- Main entry point for the application ---
if __name__ == '__main__':
    logger.info("<<< Digital Twin Welding AI System Starting >>>")

    system_manager = None
    try:
        system_manager = SystemManager()
        system_manager.start() # This will block until shutdown
    except SystemExit as e:
         logger.critical(f"System exited during initialization: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred at the top level: {e}", exc_info=True)
        if system_manager and system_manager.is_running:
            system_manager.shutdown(graceful=False)
    finally:
        # Ensure data logger is closed if it was initialized and an error occurred before graceful shutdown
        if system_manager and system_manager.data_logger and system_manager.data_logger.conn:
            logger.info("Ensuring data logger connection is closed in main finally block.")
            system_manager.data_logger.close_connection()
        logger.info("<<< Digital Twin Welding AI System Terminated >>>")