# src/system_manager.py
# Description: (MODIFIED) Manages the lifecycle and ADVANCED coordination of all system modules.
#              Includes considerations for Message Queue setup, module health, and future microservices.

import time
import logging
import threading
import signal
import datetime
import os

from src import config # Main configuration for the entire system

# Import all core modules that this manager will control
# These modules are now expected to handle their own MQ connections if USE_MESSAGE_QUEUE is True
from src.robot_control_interface import RobotControlInterface
from src.sensor_data_handler import SensorDataHandler
from src.ai_inference_engine import AIInferenceEngine
from src.quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl
from src.welding_process_manager import WeldingProcessManager
from src.data_logger_db import DataLoggerDB
from src.physics_interface import UnrealSimulatorInterface

# HMI might be a separate process communicating via API/MQ
# from src.hmi_interface import HMIInterface # Conceptual

# Setup logging (ensure config.py might have initialized a more sophisticated logger)
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') and hasattr(config.logging, 'getLogger') else logging.getLogger(__name__)
if not hasattr(config, 'logging') or not logging.getLogger(__name__).hasHandlers():
    log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - [%(levelname)s] - %(name)s - %(threadName)s - %(message)s')


class SystemManager:
    """
    (MODIFIED) Manages the overall digital twin system, focusing on module lifecycle,
    coordination (especially if not using MQ for everything), and system-level status.
    Assumes individual modules manage their own MQ connections and threads if USE_MESSAGE_QUEUE.
    """
    _instance = None # Singleton pattern
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized_globally = False # Use a different flag to avoid re-init issues with singleton
        return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized_globally:
                return
            self._initialized_globally = True

            logger.info("System Manager (Advanced with MQ considerations) initializing...")

            # --- Configuration Validation (Basic Example) ---
            self._validate_config()

            # --- Module Instances ---
            self.data_logger = None
            self.robot_interface = None
            self.sensor_handler = None
            self.ai_engine = None
            self.quality_controller = None
            self.physics_interface = None
            self.process_manager = None # Process Manager often depends on many others

            self._initialized_successfully = False
            try:
                self._initialize_all_modules() # Calls a new method for clarity
                self._initialized_successfully = True
                logger.info("All core modules initialized by System Manager successfully.")
                self.log_system_event("SYSTEM_INITIALIZED", "Core modules initialized.")
            except Exception as e:
                logger.critical(f"FATAL: System Manager failed during module initialization: {e}", exc_info=True)
                self.log_system_event("SYSTEM_INITIALIZATION_FAILED", f"Error: {e}")
                # Attempt cleanup of any partially initialized modules, especially DataLoggerDB
                if self.data_logger and hasattr(self.data_logger, 'close_connection'):
                     logger.info("Attempting to close DataLoggerDB connection after failed init.")
                     self.data_logger.close_connection()
                # Further cleanup logic might be needed depending on module __init__ side effects

            self.is_running = False
            self._shutdown_event = threading.Event()

            if self._initialized_successfully:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                logger.info("Signal handlers registered for graceful shutdown.")
            else:
                logger.critical("System initialization failed. System will not start. Signal handlers not registered.")


    def _validate_config(self):
        """Validates essential configuration parameters from config.py."""
        logger.info("Validating essential configurations...")
        required_attrs = {
            'DB_TYPE': None, # Check existence, value check is up to DataLoggerDB
            'MQ_TYPE': None, # Check existence if USE_MESSAGE_QUEUE
            'UE_SIMULATOR_IP': None,
            'UE_SIMULATOR_PORT': None,
            'ROBOT_CONFIGS': [], # Expect a list, possibly empty
            'SENSOR_CONFIGS': [], # Expect a list, possibly empty
            'MODEL_SAVE_PATH': None,
            'SCALER_SAVE_PATH': None,
            'LOG_LEVEL': 'INFO', # Default if not present
        }
        if getattr(config, 'USE_MESSAGE_QUEUE', False): # Only check MQ if it's supposed to be used
            required_attrs.update({
                'MQ_HOST': None, 'MQ_PORT': None,
                'MQ_SENSOR_DATA_EXCHANGE': None,
                'MQ_AI_PREDICTION_EXCHANGE': None,
                # Add other essential MQ exchange/queue names
            })

        missing = []
        for attr, default_val in required_attrs.items():
            if not hasattr(config, attr):
                if default_val is not None and attr not in ['ROBOT_CONFIGS', 'SENSOR_CONFIGS']: # Allow empty lists
                    logger.warning(f"Config attribute '{attr}' not found. Using default: '{default_val}' if applicable.")
                    # setattr(config, attr, default_val) # Optionally set default in config module itself
                elif default_val is None: # Required attribute
                    missing.append(attr)
        if missing:
            logger.critical(f"FATAL: Essential config attributes missing: {', '.join(missing)}. System cannot start reliably.")
            raise ValueError(f"Missing essential configurations: {', '.join(missing)}")
        logger.info("Essential configurations validated.")


    def _initialize_all_modules(self):
        """Initializes all core system modules in specified order."""
        logger.info("Initializing core system modules in sequence...")

        # 1. Data Logger (Needed early for other modules to log during their init)
        if getattr(config, 'ENABLE_DATA_LOGGING', True):
            self.data_logger = DataLoggerDB() # Singleton, config read inside
            if not self.data_logger._initialized or (self.data_logger.db_type != 'sqlite' and self.data_logger.conn is None): # Check successful init
                 raise ConnectionError("DataLoggerDB failed to initialize or connect. Halting SystemManager init.")
            logger.info("Data Logger DB initialized by SystemManager.")
        else:
            logger.warning("Data Logging is disabled in config.")

        # 2. Physics/Visualization Interface (Needed by ProcessManager)
        if getattr(config, 'ENABLE_PHYSICS_INTERFACE', True):
            self.physics_interface = UnrealSimulatorInterface() # Singleton, config read inside
            # Connection happens in start()
            logger.info("Physics/Visualization Interface initialized by SystemManager.")
        else:
            logger.warning("Physics Interface is disabled in config.")


        # 3. Robot Control Interface (Needed by ProcessManager)
        if getattr(config, 'ENABLE_ROBOT_INTERFACE', True):
            self.robot_interface = RobotControlInterface(data_logger=self.data_logger) # Singleton, config read inside
            # Connection and monitoring threads start in its own start_interface()
            logger.info("Robot Control Interface initialized by SystemManager.")
        else:
            logger.warning("Robot Interface is disabled in config.")


        # 4. Sensor Data Handler (Publishes to MQ, needed by AI Engine, QEA)
        if getattr(config, 'ENABLE_SENSOR_HANDLER', True):
            self.sensor_handler = SensorDataHandler(data_logger=self.data_logger) # Singleton
            # Collection and MQ publishing threads start in its own start_collection()
            logger.info("Sensor Data Handler initialized by SystemManager.")
        else:
            logger.warning("Sensor Data Handler is disabled in config.")


        # 5. AI Inference Engine (Subscribes to Sensor MQ, Publishes to Prediction MQ)
        if getattr(config, 'ENABLE_AI_INFERENCE', True):
            self.ai_engine = AIInferenceEngine(data_logger=self.data_logger) # Singleton
            if not self.ai_engine.model_loaded:
                 logger.warning("AI Model not loaded in AIInferenceEngine. Predictions may be unavailable.")
            # MQ consumption threads start in its own start_consuming_sensor_data()
            logger.info("AI Inference Engine initialized by SystemManager.")
        else:
            logger.warning("AI Inference Engine is disabled in config.")


        # 6. Quality Evaluator and Adaptive Control (Subscribes to Prediction MQ, Publishes Control MQ)
        if getattr(config, 'ENABLE_ADAPTIVE_CONTROL', True):
            self.quality_controller = QualityEvaluatorAdaptiveControl(
                data_logger=self.data_logger,
                sensor_handler=self.sensor_handler, # For direct sensor access if needed beyond MQ
                physics_interface=self.physics_interface # For physics sim results if used in ensemble
            ) # Singleton
            # MQ consumption/worker threads start in its own start() or start_consuming_predictions()
            logger.info("Quality Evaluator & Adaptive Control initialized by SystemManager.")
        else:
            logger.warning("Quality Evaluator & Adaptive Control is disabled in config.")


        # 7. Welding Process Manager (Main orchestrator, uses MQ for many interactions)
        self.process_manager = WeldingProcessManager(
            data_logger=self.data_logger,
            physics_interface=self.physics_interface, # For direct UE viz commands
            robot_control_interface_direct_access=self.robot_interface # For direct pose polling if needed
        ) # Singleton
        # MQ consumption/worker threads start in its own start()
        logger.info("Welding Process Manager initialized by SystemManager.")

        # 8. HMI Interface (Conceptual - if managed by SystemManager)
        # if getattr(config, 'ENABLE_HMI', False) and HMIInterface:
        #     self.hmi_interface = HMIInterface(self, ...) # Pass SystemManager instance for control
        #     logger.info("HMI Interface initialized by SystemManager.")


    def log_system_event(self, event_type, message, details=None):
        # ... (Same as previous version, with check for self.data_logger and its connection) ...
        if self.data_logger and hasattr(self.data_logger, 'conn') and self.data_logger.conn is not None:
            try:
                # ... (logging logic) ...
                pass
            except Exception as e: logger.error(f"Failed to log system event via DB: {e}")
        else: logger.warning(f"DataLogger not available for system event: {event_type} - {message}")


    def start(self):
        """Starts all managed services and the main system loop."""
        if not self._initialized_successfully:
             logger.critical("System Manager failed to initialize. Cannot start. Please check previous logs.")
             raise SystemExit("System initialization failed.")
        if self.is_running: logger.warning("System Manager is already running."); return

        logger.info("Starting System Manager and core services...")
        self.is_running = True
        self._shutdown_event.clear()
        self.log_system_event("SYSTEM_STARTUP", "System Manager starting services.")

        # --- Start Core Services/Threads (Modules handle their own internal threads and MQ connections) ---
        # SystemManager ensures each module's main start/connect method is called.

        if self.physics_interface and hasattr(self.physics_interface, 'start_connection_manager'):
            self.physics_interface.start_connection_manager() # Starts its bg connection/reconnection
            # Actual connect attempt happens in its own thread. We don't block here.
            logger.info("Physics Interface connection manager started.")

        if self.robot_interface and hasattr(self.robot_interface, 'start_interface'):
            self.robot_interface.start_interface() # Connects to robots, starts MQ command consumer & status publisher
            logger.info("Robot Control Interface services started.")

        if self.sensor_handler and hasattr(self.sensor_handler, 'start_collection'):
            self.sensor_handler.start_collection() # Starts sensor collectors & MQ publisher
            logger.info("Sensor Data Handler collection started.")

        if self.ai_engine and hasattr(self.ai_engine, 'start_consuming_sensor_data'):
            self.ai_engine.start_consuming_sensor_data() # Starts MQ sensor consumer & inference worker
            logger.info("AI Inference Engine MQ consumer started.")

        if self.quality_controller and hasattr(self.quality_controller, 'start'): # Assuming QEA has a start method
            self.quality_controller.start() # Starts MQ prediction consumer & evaluation worker
            logger.info("Quality Evaluator & Adaptive Control services started.")

        if self.process_manager and hasattr(self.process_manager, 'start'): # Assuming WPM has a start method
            self.process_manager.start() # Starts its MQ response consumer
            logger.info("Welding Process Manager services started.")

        # if self.hmi_interface and hasattr(self.hmi_interface, 'start_server'):
        #     self.hmi_interface.start_server() # Start HMI server if it's part of this process

        logger.info("System Manager is now RUNNING. Core services initiated.")
        self.log_system_event("SYSTEM_RUNNING", "All services initiated.")

        # --- Main Loop for System Manager (Keeps main thread alive) ---
        try:
            while not self._shutdown_event.is_set():
                # --- Perform System-Level Health Checks (Example) ---
                # if self.physics_interface and not self.physics_interface.is_connected:
                #     logger.warning("Health Check: UE Interface appears disconnected.")
                #     # Potentially trigger reconnection logic if not automatic in physics_interface
                # if self.robot_interface:
                #     connected_r = self.robot_interface.get_connected_robot_ids()
                #     if len(connected_r) < len(getattr(config, 'ROBOT_CONFIGS', [])):
                #          logger.warning(f"Health Check: Not all robots connected. Expected {len(getattr(config, 'ROBOT_CONFIGS', []))}, got {len(connected_r)}.")

                # --- Handle High-Level System Commands (e.g., from a CLI, management interface) ---
                # This is distinct from HMI job commands, more like system pause/resume/reconfigure.
                # For now, this loop primarily keeps the application alive until shutdown.
                time.sleep(getattr(config, 'SYSTEM_HEALTH_CHECK_INTERVAL_SEC', 5.0))
        # ... (Exception handling and finally block as in previous SystemManager, ensuring self.shutdown() is called) ...
        except KeyboardInterrupt: logger.info("KeyboardInterrupt in SM main loop. Signal handler will shutdown.")
        except Exception as e_main: logger.critical(f"Critical error in SM main loop: {e_main}", exc_info=True); self.shutdown(graceful=False)
        finally:
             if self.is_running or not self._shutdown_event.is_set(): # If loop exited unexpectedly
                  logger.warning("SM main loop exited unexpectedly. Initiating shutdown."); self.shutdown(graceful=False)
        logger.info("System Manager main loop finished.")


    def shutdown(self, graceful=True):
        if self._shutdown_event.is_set() and not self.is_running: logger.info("Shutdown already in progress or completed."); return
        logger.info(f"System Manager: Initiating {'graceful' if graceful else 'forced'} system shutdown...")
        self._shutdown_event.set()
        self.is_running = False
        self.log_system_event("SYSTEM_SHUTDOWN_INITIATED", f"Graceful: {graceful}")

        # Stop modules in an order that minimizes issues (e.g., stop consumers before producers if possible)
        # Or stop high-level orchestrators first.
        if self.process_manager and hasattr(self.process_manager, 'stop'): self.process_manager.stop()
        if self.quality_controller and hasattr(self.quality_controller, 'stop'): self.quality_controller.stop()
        if self.ai_engine and hasattr(self.ai_engine, 'stop_consuming_sensor_data'): self.ai_engine.stop_consuming_sensor_data()
        if self.sensor_handler and hasattr(self.sensor_handler, 'stop_collection'): self.sensor_handler.stop_collection()
        if self.robot_interface and hasattr(self.robot_interface, 'stop_interface'): self.robot_interface.stop_interface()
        if self.physics_interface and hasattr(self.physics_interface, 'disconnect'): self.physics_interface.disconnect()
        # if self.hmi_interface and hasattr(self.hmi_interface, 'stop_server'): self.hmi_interface.stop_server()

        # DataLogger should be last if other modules might log during their shutdown
        if self.data_logger and hasattr(self.data_logger, 'close_connection'):
            logger.info("Closing Data Logger connection as part of system shutdown.")
            self.data_logger.close_connection()

        logger.info("System Manager shutdown sequence complete.")
        self.log_system_event("SYSTEM_SHUTDOWN_COMPLETE", "All services signaled to stop.")


    def _signal_handler(self, signum, frame):
        # ... (Same as previous version) ...
        pass # For brevity

    # --- External Control Methods (start_job_externally, stop_job_externally, get_system_status_externally) ---
    # These remain largely the same, delegating to self.process_manager or providing status.
    # get_system_status_externally should be updated to reflect MQ status and module health.
    def get_system_status_externally(self):
        # ... (Previous version, but now add MQ connection status from each relevant module if available) ...
        # ... (e.g., self.robot_interface.get_mq_status(), self.ai_engine.get_mq_status()) ...
        # This requires each module to expose its MQ status.
        status = {
            "manager_status": "RUNNING" if self.is_running else ("SHUTTING_DOWN" if self._shutdown_event.is_set() else ("INITIALIZATION_FAILED" if not self._initialized_successfully else "STOPPED")),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "process_manager": self.process_manager.get_manager_status() if self.process_manager and hasattr(self.process_manager, 'get_manager_status') else None,
            "robot_interface": self.robot_interface.get_status() if self.robot_interface and hasattr(self.robot_interface, 'get_status') else None, # Assume get_status() added to RCI
            "physics_interface": {"connected": self.physics_interface.is_connected} if self.physics_interface else None,
            "ai_engine": self.ai_engine.get_status() if self.ai_engine and hasattr(self.ai_engine, 'get_status') else None,
            "sensor_handler": self.sensor_handler.get_status() if self.sensor_handler and hasattr(self.sensor_handler, 'get_status') else None,
            "quality_evaluator": self.quality_controller.get_status() if self.quality_controller and hasattr(self.quality_controller, 'get_status') else None, # Assume get_status() added
            "data_logger": {"connected": self.data_logger.conn is not None} if self.data_logger and hasattr(self.data_logger, 'conn') else None,
        }
        return status
    # ... (start_job_externally and stop_job_externally remain similar, delegating to self.process_manager) ...


# --- Main entry point ---
if __name__ == '__main__':
    # ... (Main execution block similar to previous version, ensuring SystemManager init and start) ...
    logger.info("<<< Digital Twin Welding AI System (Advanced MQ) Starting >>>")
    system_manager_instance = None
    try:
        system_manager_instance = SystemManager()
        if system_manager_instance._initialized_successfully:
            system_manager_instance.start() # This blocks until shutdown
        else:
            logger.critical("System Manager did not initialize successfully. Exiting.")
    # ... (Exception handling and finally block similar to previous, ensuring shutdown/cleanup) ...
    except SystemExit as e: logger.critical(f"System exited: {e}")
    except Exception as e_top: logger.critical(f"Top-level unhandled exception: {e_top}", exc_info=True)
    finally:
        if system_manager_instance and (system_manager_instance.is_running or not system_manager_instance._shutdown_event.is_set()):
            logger.warning("SystemManager exiting main block, ensuring shutdown is called.")
            try: system_manager_instance.shutdown(graceful=False)
            except Exception as e_final_shutdown: logger.critical(f"Error during final shutdown attempt: {e_final_shutdown}")
        logger.info("<<< Digital Twin Welding AI System (Advanced MQ) Terminated >>>")