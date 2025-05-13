# src/system_manager.py
# Description: Manages the lifecycle and coordination of all system modules.
#              Initializes, starts, stops, and monitors core components.

import time
import logging
import threading # For managing module threads or background tasks
import signal # For handling system shutdown signals
import datetime # For timestamps in logging/status

import config # Main configuration for the entire system
# Import all core modules that this manager will control
from robot_control_interface import RobotControlInterface
from sensor_data_handler import SensorDataHandler
from ai_inference_engine import AIInferenceEngine
from quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl
from welding_process_manager import WeldingProcessManager
from data_logger_db import DataLoggerDB
from physics_interface import UnrealSimulatorInterface # Import the physics interface

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
            logger.warning("ROBOT_CONFIGS not found in config. Robot control interface may be non-functional.")
        if not hasattr(config, 'SENSOR_CONFIGS'):
             logger.warning("SENSOR_CONFIGS not found in config. Sensor data collection may be non-functional.")
        if not hasattr(config, 'DATABASE_PATH'):
             logger.warning("DATABASE_PATH not found in config. Data logging might use default or fail.")
        if not hasattr(config, 'UE_SIMULATOR_IP') or not hasattr(config, 'UE_SIMULATOR_PORT'):
             logger.warning("UE_SIMULATOR_IP or PORT not found in config. Physics/Visualization interface may be non-functional.")


        # --- Module Instances ---
        self.data_logger = None
        self.robot_interface = None
        self.sensor_handler = None
        self.ai_engine = None
        self.quality_controller = None
        self.process_manager = None
        self.physics_interface = None # Add physics_interface instance
        # self.hmi_interface = None

        # Flag to track successful initialization
        self._initialized_successfully = False

        try:
            self._initialize_modules()
            self._initialized_successfully = True
            logger.info("All core modules initialized successfully.")
            self.log_system_event("SYSTEM_INITIALIZED", "All modules initialized successfully.")

        except Exception as e:
            logger.critical(f"Fatal error during module initialization: {e}", exc_info=True)
            self.log_system_event("SYSTEM_INITIALIZATION_FAILED", f"Fatal error during initialization: {e}")
            # Attempt to clean up already initialized modules before exiting
            # A more robust approach would track initialized modules and clean them up individually
            if self.data_logger: self.data_logger.close_connection() # Close DB connection if it was opened


        self.is_running = False
        self._shutdown_event = threading.Event() # Event to signal shutdown

        # Register signal handlers for graceful shutdown, only if initialized
        if self._initialized_successfully:
            signal.signal(signal.SIGINT, self._signal_handler)  # Handle Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler) # Handle `kill` command
            logger.info("Signal handlers registered for graceful shutdown.")
        else:
            logger.critical("System initialization failed. Signal handlers not registered. Immediate exit on interrupt.")


    def _initialize_modules(self):
        """Initializes all core system modules."""
        logger.info("Initializing core system modules...")

        # 1. Data Logger (often needed by other modules for logging during init)
        # DataLoggerDB is a singleton, subsequent calls get the same instance.
        self.data_logger = DataLoggerDB()
        logger.info("Data Logger DB initialized.")
        # Check if DB connection was successful during its init
        if self.data_logger.conn is None:
            raise ConnectionError("Failed to connect to database.")


        # 2. Physics/Visualization Interface (needed by ProcessManager)
        self.physics_interface = UnrealSimulatorInterface()
        logger.info("Physics/Visualization Interface initialized.")

        # 3. Robot Control Interface (needed by ProcessManager)
        self.robot_interface = RobotControlInterface() # Reads ROBOT_CONFIGS from config
        logger.info("Robot Control Interface initialized.")

        # 4. Sensor Data Handler (needed by ProcessManager and AI Engine potentially)
        # Pass data_logger instance if sensor_handler logs data directly upon collection
        self.sensor_handler = SensorDataHandler(data_logger=self.data_logger)
        logger.info("Sensor Data Handler initialized.")

        # 5. AI Inference Engine (needed by QualityController)
        self.ai_engine = AIInferenceEngine() # Loads model and scaler from config paths during its init
        # AI Engine initialization might fail if model/scaler files are missing.
        # Decide if this should be a fatal error or just disable AI inference.
        # Current AIInferenceEngine handles missing files internally by setting model_loaded/scaler_loaded flags.
        logger.info("AI Inference Engine initialized.")
        if not self.ai_engine.model_loaded:
             logger.warning("AI Model not loaded successfully. AI prediction will be unavailable.")
             self.log_system_event("AI_MODEL_NOT_LOADED", "AI Model file missing or failed to load.")


        # 6. Quality Evaluator and Adaptive Control (depends on AI Engine, Sensor Handler)
        # Pass necessary dependencies
        self.quality_controller = QualityEvaluatorAdaptiveControl(
            # Pass dependencies QualityEvaluatorAdaptiveControl needs if its __init__ changes
            # e.g., ai_engine=self.ai_engine, sensor_handler=self.sensor_handler
        ) # Uses config for rules/thresholds
        logger.info("Quality Evaluator & Adaptive Control initialized.")


        # 7. Welding Process Manager (depends on many of the above)
        # Pass ALL dependencies ProcessManager needs to orchestrate
        self.process_manager = WeldingProcessManager(
            robot_interface=self.robot_interface,
            ai_engine=self.ai_engine,
            quality_controller=self.quality_controller,
            sensor_handler=self.sensor_handler,
            # Pass the physics interface instance to WPM
            physics_interface=self.physics_interface, # <-- Pass physics_interface here
            # Pass data_logger if WPM logs directly, or WPM passes data to logger
            data_logger=self.data_logger # Pass DataLoggerDB instance
        )
        logger.info("Welding Process Manager initialized.")

        # 8. HMI Interface (Placeholder)
        # self.hmi_interface = HMIInterface(self.process_manager, self.data_logger, self) # Pass manager, logger, maybe SystemManager itself
        # logger.info("HMI Interface initialized (Placeholder).")

        logger.info("Core module initialization process complete.")


    def log_system_event(self, event_type, message, details=None):
        """Helper to log system-level events via DataLoggerDB."""
        # Ensure data_logger is initialized before attempting to log
        if self.data_logger and self.data_logger.conn is not None:
            try:
                log_details = {"message": message}
                if details:
                    log_details.update(details)
                self.data_logger.log_process_event(event_type="SYSTEM_" + event_type, details=log_details)
            except Exception as e:
                # Fallback logging if DB logging fails
                logger.error(f"Failed to log system event '{event_type}' to DB: {e}")
                logger.error(f"Event details: {message} | {details}")
        else:
            # Use standard logger if data_logger is not initialized or connected
            logger.warning(f"DataLogger not available. System Event not logged to DB: {event_type} - {message}")


    def start(self):
        """Starts all managed services and the main system loop."""
        if not self._initialized_successfully:
             logger.critical("System Manager failed to initialize. Cannot start.")
             raise SystemExit("System initialization failed.")

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
            connected_robots = self.robot_interface.connect_all()
            if connected_robots:
                 logger.info(f"Successfully connected to robots: {connected_robots}")
                 self.log_system_event("ROBOT_CONNECTION", "Robots connected.", {"connected_ids": connected_robots})
            else:
                 logger.warning("Failed to connect to any robots. System may operate in a limited mode.")
                 self.log_system_event("ROBOT_CONNECTION_FAILED", "No robots connected.")

        # 2. Connect to Unreal Engine Simulator/Visualizer
        if self.physics_interface:
            logger.info("Connecting to Unreal Engine simulator/visualizer...")
            if self.physics_interface.connect():
                logger.info("Successfully connected to Unreal Engine.")
                self.log_system_event("UE_CONNECTION", "Connected to Unreal Engine.")
            else:
                logger.warning("Failed to connect to Unreal Engine. Visualization/Physics simulation may be unavailable.")
                self.log_system_event("UE_CONNECTION_FAILED", "Failed to connect to Unreal Engine.")

        # 3. Start Sensor Data Handler (if it has a start method for collection threads)
        if self.sensor_handler and hasattr(self.sensor_handler, 'start_collection'):
            logger.info("Starting sensor data collection...")
            try:
                self.sensor_handler.start_collection()
                self.log_system_event("SENSOR_HANDLER_START", "Sensor data collection started.")
            except Exception as e:
                logger.error(f"Failed to start sensor data collection: {e}", exc_info=True)
                self.log_system_event("SENSOR_HANDLER_ERROR", f"Failed to start collection: {e}")

        # 4. Start HMI Interface (if it runs its own server/loop)
        # if self.hmi_interface and hasattr(self.hmi_interface, 'start'):
        #     logger.info("Starting HMI interface...")
        #     self.hmi_interface.start() # Hypothetical start method


        logger.info("System Manager is now running. Waiting for external triggers (e.g., HMI) or internal jobs.")
        # --- Main loop for System Manager ---
        # This loop keeps the main thread alive and can handle high-level tasks
        # like monitoring other modules, processing external commands, etc.
        # Most core tasks (sensor collection, process execution) should run in their own threads.
        try:
            while not self._shutdown_event.is_set():
                # --- System Health Monitoring (Example) ---
                # Periodically check if essential connections (Robots, UE, DB) are alive
                # if self.robot_interface and not self.robot_interface.is_healthy(): # Hypothetical health check
                #     logger.warning("Robot Interface unhealthy!")
                #     self.log_system_event("HEALTH_CHECK_WARNING", "Robot Interface unhealthy.")
                # if self.physics_interface and not self.physics_interface.is_connected:
                #     logger.warning("UE Interface disconnected!")
                #     self.log_system_event("HEALTH_CHECK_WARNING", "UE Interface disconnected.")
                #     # Optionally try to reconnect: self.physics_interface.connect()

                # --- Handle External Commands (Placeholder) ---
                # This logic receives commands (like start_job, stop_job) from HMI/API
                # and calls the appropriate methods on process_manager, etc.
                # In the Flask HMI example (hmi_application/app.py), this interaction happens
                # via direct method calls on the system_manager_instance passed to SystemAPIClient.
                # If using MQ/API, this loop would read commands from the MQ/API endpoint.

                # For now, just sleep and wait for shutdown signal or jobs started by external calls
                time.sleep(1) # Main loop heartbeat - keeps the main thread responsive

        except KeyboardInterrupt: # Caught by signal_handler
             logger.info("KeyboardInterrupt caught in main loop. Signal handler should manage shutdown.")
             # The signal handler will be called and trigger shutdown.
             # Just ensure the loop exits if signal handler wasn't registered (e.g., init failed).
             if not self._shutdown_event.is_set() and self._initialized_successfully:
                  logger.error("KeyboardInterrupt received but shutdown not flagged. Forcing shutdown.")
                  self.shutdown(graceful=False) # Fallback force shutdown

        except Exception as e:
            logger.critical(f"Unhandled exception in System Manager main loop: {e}", exc_info=True)
            self.log_system_event("SYSTEM_CRITICAL_ERROR", f"Unhandled exception in main loop: {e}")
            # Attempt graceful shutdown on error, but allow force if graceful fails
            try:
                 self.shutdown(graceful=True)
            except Exception as shutdown_e:
                 logger.error(f"Error during graceful shutdown after main loop error: {shutdown_e}")
                 self.shutdown(graceful=False) # Force shutdown


        # The loop exited (either shutdown_event was set or exception occurred).
        # The shutdown() method should have been called by signal_handler or the exception handler.
        # If for some reason shutdown wasn't called, call it now as a final attempt.
        if self.is_running or not self._shutdown_event.is_set():
             logger.warning("Main loop exited, but shutdown was not fully initiated. Attempting shutdown.")
             self.shutdown(graceful=False) # Force shutdown if graceful wasn't started

        logger.info("System Manager main loop finished.")
        # System will exit after this if this is the main thread.


    def shutdown(self, graceful=True):
        """Shuts down all managed services and modules."""
        # Use _shutdown_event to prevent multiple shutdown sequences from running concurrently
        if self._shutdown_event.is_set() and self.is_running == False:
            logger.info("Shutdown already initiated or completed.")
            return # Already in shutdown process

        logger.info(f"Initiating {'graceful' if graceful else 'forced'} system shutdown...")
        # Set shutdown event first to signal all dependent threads/loops
        self._shutdown_event.set()
        # Mark system as not running
        self.is_running = False # Should be set by the loop exiting, but set explicitly here too

        self.log_system_event("SYSTEM_SHUTDOWN_START", f"Graceful: {graceful}")

        # --- Stop Core Services/Threads (in reverse order of start or based on dependencies) ---
        # Dependencies: WPM -> (AI, Quality, Sensor, Robot, Physics). Sensor -> Logger. Physics -> Logger. Robot -> Logger.
        # Stop in reverse order of dependency or in order of risk/externality:
        # WPM (stops jobs, stops AC loops), Sensor Handler (stops collection threads),
        # Robot Interface (disconnects robots), Physics Interface (disconnects UE),
        # HMI Interface (stops server), Data Logger (closes DB connection).

        # 1. Stop Welding Process Manager (tells it to stop any active jobs and its threads)
        if self.process_manager:
            logger.info("Stopping Welding Process Manager...")
            try:
                # stop_current_job signals WPM's internal threads/loops to stop and sends robot stop commands.
                self.process_manager.stop_current_job()
                # Wait briefly for WPM's main job thread to complete after being signaled.
                if hasattr(self.process_manager, 'process_thread') and self.process_manager.process_thread and self.process_manager.process_thread.is_alive():
                     logger.info("Waiting for Process Manager job thread to complete...")
                     # Use a reasonable timeout. If graceful is False, use a shorter timeout.
                     wait_timeout = 10 if graceful else 2
                     self.process_manager.process_thread.join(timeout=wait_timeout)
                     if self.process_manager.process_thread.is_alive():
                          logger.warning("Process Manager job thread did not terminate gracefully.")
                     else:
                         logger.info("Process Manager job thread terminated.")
            except Exception as e:
                logger.error(f"Error stopping Welding Process Manager: {e}", exc_info=True)


        # 2. Stop Sensor Data Handler (stops its collection and processing threads)
        if self.sensor_handler and hasattr(self.sensor_handler, 'stop_collection'):
            logger.info("Stopping sensor data collection...")
            try:
                self.sensor_handler.stop_collection() # Hypothetical stop method
                self.log_system_event("SENSOR_HANDLER_STOP", "Sensor data collection stopped.")
            except Exception as e:
                 logger.error(f"Error stopping sensor data collection: {e}", exc_info=True)


        # 3. Disconnect from Robots (stops monitoring threads within interface if any)
        if self.robot_interface:
            logger.info("Disconnecting from robots...")
            try:
                self.robot_interface.disconnect_all()
                self.log_system_event("ROBOT_DISCONNECTION", "Robots disconnected.")
            except Exception as e:
                 logger.error(f"Error disconnecting robots: {e}", exc_info=True)

        # 4. Disconnect from Unreal Engine Simulator/Visualizer
        if self.physics_interface:
            logger.info("Disconnecting from Unreal Engine simulator/visualizer...")
            try:
                self.physics_interface.disconnect()
                self.log_system_event("UE_DISCONNECTION", "Unreal Engine disconnected.")
            except Exception as e:
                 logger.error(f"Error disconnecting Unreal Engine: {e}", exc_info=True)

        # 5. Stop HMI Interface (if it runs its own server/loop in this process)
        # if self.hmi_interface and hasattr(self.hmi_interface, 'stop'):
        #     logger.info("Stopping HMI interface...")
        #     try:
        #          self.hmi_interface.stop()
        #     except Exception as e:
        #          logger.error(f"Error stopping HMI interface: {e}", exc_info=True)

        # 6. Close Data Logger connection (should be one of the last steps)
        # Ensure the connection is closed even if other steps failed.
        if self.data_logger:
            logger.info("Closing Data Logger connection...")
            try:
                self.data_logger.close_connection()
                # No DB event logging possible after this point
            except Exception as e:
                 logger.error(f"Error closing data logger: {e}", exc_info=True)


        logger.info("System Manager shutdown complete.")
        self.log_system_event("SYSTEM_SHUTDOWN_COMPLETE", "System Manager shutdown process finished.")
        # The main thread will exit shortly after this.


    def _signal_handler(self, signum, frame):
        """Handles OS signals like SIGINT (Ctrl+C) and SIGTERM."""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"Received signal {signal_name}. Initiating graceful shutdown...")
        # Set the shutdown event. The main loop and child threads should detect this.
        # Call the shutdown method to perform cleanup.
        # Use a separate thread for shutdown if the signal handler needs to return quickly,
        # but for simple cleanup like ours, calling directly might be sufficient unless
        # the cleanup itself is blocking or prone to deadlocks in signal context.
        # Direct call for simplicity:
        if not self._shutdown_event.is_set(): # Prevent re-entry if already shutting down
             self.shutdown(graceful=True)
        else:
             logger.info("Shutdown already in progress. Ignoring signal.")

    # --- Methods for external control (e.g., from HMI or API) ---
    # These methods are called by the HMI client (SystemAPIClient in app.py)
    # They should be thread-safe if the HMI runs in a different thread or process
    # and interacts with the SystemManager instance's methods directly.
    # The SystemAPIClient in app.py handles the direct call vs remote call abstraction.

    def start_job_externally(self, job_id):
        """Allows an external system to request a job start."""
        if not self.is_running:
            logger.error(f"Cannot start job '{job_id}': System Manager is not running.")
            self.log_system_event("EXTERNAL_JOB_START_REJECTED", f"Manager not running. Job: {job_id}")
            return False
        # Delegate the request to the Process Manager
        if self.process_manager:
            logger.info(f"External request received: Start Job '{job_id}'")
            success = self.process_manager.start_welding_job(job_id)
            self.log_system_event("EXTERNAL_JOB_START_REQUEST", f"Job: {job_id}, Success: {success}")
            return success
        else:
            logger.error("Process Manager not available to start job.")
            self.log_system_event("EXTERNAL_JOB_START_REJECTED", f"Process Manager not available. Job: {job_id}")
            return False

    def stop_job_externally(self):
        """Allows an external system to request stopping the current job."""
        if not self.is_running:
            logger.error("Cannot stop job: System Manager is not running.")
            self.log_system_event("EXTERNAL_JOB_STOP_REJECTED", "Manager not running.")
            return False
        # Delegate the request to the Process Manager
        if self.process_manager:
            logger.info("External request received: Stop Current Job.")
            success = self.process_manager.stop_current_job()
            self.log_system_event("EXTERNAL_JOB_STOP_REQUEST", f"Success: {success}")
            return success
        else:
            logger.error("Process Manager not available to stop job.")
            self.log_system_event("EXTERNAL_JOB_STOP_REJECTED", "Process Manager not available.")
            return False

    def get_system_status_externally(self):
        """Provides a summary of the system status for external queries."""
        # Check if core components were initialized. If not, return a basic error status.
        if not self._initialized_successfully:
             return {
                 "manager_status": "INITIALIZATION_FAILED",
                 "error": "System Manager failed to initialize its core modules.",
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
             }

        # Gather status from key components
        status = {
            "manager_status": "RUNNING" if self.is_running else ("SHUTTING_DOWN" if self._shutdown_event.is_set() else "STOPPED"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            # Delegate status requests to individual managers/interfaces
            "process_manager_status": self.process_manager.get_manager_status() if self.process_manager else None,
            "robot_interface_status": {
                "is_initialized": self.robot_interface is not None,
                "is_connected": self.robot_interface.is_connected if self.robot_interface else False,
                "connected_robots": self.robot_interface.connected_robot_ids if self.robot_interface else []
            },
            "physics_interface_status": { # Include physics interface status
                 "is_initialized": self.physics_interface is not None,
                 "is_connected": self.physics_interface.is_connected if self.physics_interface else False
            },
            "ai_engine_status": {
                "is_initialized": self.ai_engine is not None,
                "model_loaded": self.ai_engine.model_loaded if self.ai_engine else False,
                "scaler_loaded": self.ai_engine.scaler_loaded if self.ai_engine else False
            },
            "sensor_handler_status": self.sensor_handler.get_status() if self.sensor_handler and hasattr(self.sensor_handler, 'get_status') else "UNKNOWN_OR_NO_GET_STATUS",
            "data_logger_status": {
                 "is_initialized": self.data_logger is not None,
                 "is_connected": self.data_logger.conn is not None # Check DB connection
            },
            # Add status from other modules (e.g., HMI interface status if it's managed here)
        }
        # logger.debug("Provided system status externally.") # Too verbose for frequent polling
        return status


# --- Main entry point for the application ---
# When system_manager.py is run directly, this block executes.
if __name__ == '__main__':
    logger.info("<<< Digital Twin Welding AI System Starting >>>")

    system_manager = None
    try:
        # 1. Initialize the SystemManager and all its modules
        logger.info("Attempting to initialize SystemManager...")
        system_manager = SystemManager()
        logger.info("SystemManager initialization phase completed.")

        # 2. Start the SystemManager services (connections, background threads/loops)
        # This method contains the main loop and will block until shutdown.
        logger.info("Attempting to start SystemManager...")
        system_manager.start()
        logger.info("SystemManager start method returned.") # This line is reached after shutdown_event is set

    except SystemExit as e:
         logger.critical(f"System exited during initialization or startup: {e}")
         # The SystemManager __init__ or start method should handle cleanup before raising SystemExit
    except Exception as e:
        # Catch any unhandled exceptions that occur BEFORE the main loop is properly running
        # or that somehow escape the main loop's internal handling.
        logger.critical(f"An unhandled exception occurred at the top level: {e}", exc_info=True)
        # Attempt to shut down gracefully, or force if needed
        if system_manager and system_manager.is_running: # If start() was called and running flag is True
             logger.warning("Attempting graceful shutdown from top-level exception handler.")
             try:
                 system_manager.shutdown(graceful=True)
             except Exception as shutdown_e:
                 logger.error(f"Error during graceful shutdown attempt after exception: {shutdown_e}")
                 logger.warning("Attempting forced shutdown.")
                 if system_manager:
                      try:
                           system_manager.shutdown(graceful=False)
                      except Exception as forced_shutdown_e:
                           logger.critical(f"FATAL: Error during forced shutdown after exception: {forced_shutdown_e}")
        elif system_manager and system_manager._initialized_successfully:
             # If initialized but start() wasn't fully successful or running==False
             logger.warning("SystemManager initialized but not running. Attempting cleanup.")
             try:
                 system_manager.shutdown(graceful=False) # Force cleanup of initialized modules
             except Exception as cleanup_e:
                  logger.critical(f"FATAL: Error during cleanup after exception: {cleanup_e}")
        else:
             # If initialization failed entirely, basic cleanup might be needed (e.g. if DB connection opened before error)
             logger.critical("SystemManager failed to initialize. No instance available for shutdown.")
             # Manual cleanup for critical resources initialized early, e.g., DataLoggerDB singleton
             try:
                 # Get the singleton instance if it was created
                 data_logger_instance = DataLoggerDB(db_path=getattr(config, 'DATABASE_PATH', None))
                 if data_logger_instance and data_logger_instance.conn:
                      logger.warning("Attempting manual DataLoggerDB cleanup.")
                      data_logger_instance.close_connection()
             except Exception as manual_cleanup_e:
                  logger.critical(f"FATAL: Error during manual DataLoggerDB cleanup: {manual_cleanup_e}")


    finally:
        # This finally block is executed when the main thread is about to exit.
        # If SystemManager.start() ran its main loop, the shutdown() method should
        # have been called already by the signal handler or the exception handler.
        # This block is primarily useful if the script exits prematurely *before*
        # the start() method's loop is entered or before signal handlers are effective.
        # Redundant check: Ensure data logger is closed if it was initialized and not closed by shutdown.
        # The shutdown method itself contains the logger closing logic.
        # A flag could be used to ensure shutdown isn't called multiple times.
        pass # Let the shutdown method handle final cleanup

    logger.info("<<< Digital Twin Welding AI System Terminated >>>")