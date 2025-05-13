# hmi_application/app.py
# Description: Flask web application for the Digital Twin Welding AI System HMI.
# Provides a web interface to monitor system status and control jobs.

from flask import Flask, render_template, request, redirect, url_for, jsonify
import sys
import os
import logging
import datetime
import threading
import argparse

# Setup basic logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Step 1: Add project root directory to sys.path ---
# This allows importing modules from the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at beginning to ensure our modules take precedence
    logger.info(f"Added project root '{project_root}' to sys.path.")
else:
    logger.info(f"Project root '{project_root}' already in sys.path.")

# --- Step 2: Attempt to import SystemManager and core dependencies ---
SystemManager = None
system_manager_available = False
try:
    # Import config first as other modules depend on it
    from src.config import Config
    logger.info("Successfully imported Config module.")

    # Import core components
    from src.data_logger_db import DataLoggerDB
    from src.robot_control_interface import RobotControlInterface
    from src.ai_inference_engine import AIInferenceEngine
    from src.sensor_data_handler import SensorDataHandler
    from src.welding_process_manager import WeldingProcessManager
    from src.system_manager import SystemManager

    logger.info("Successfully imported SystemManager and core modules.")
    system_manager_available = True

except ImportError as e:
    logger.critical(f"FATAL ERROR: Could not import SystemManager or its dependencies: {e}", exc_info=True)
    logger.critical("SystemManager will not be available. HMI will use a mock client.")
    SystemManager = None


# --- Step 3: Define SystemAPIClient ---
# This client abstraction allows the Flask app to interact with the SystemManager.
# It can interact with a real SystemManager instance (if available) or act as a mock.
class SystemAPIClient:
    """
    Client to interact with the SystemManager process.
    In a real system, this would use HTTP requests, MQ, RPC, etc., to communicate
    with a separate SystemManager process. For this demo, it directly calls methods
    of a SystemManager instance if running in the same process, or returns mock data.
    """
    def __init__(self, manager_instance=None):
        self._manager = manager_instance # A real SystemManager instance for same-process demo
        if manager_instance:
            logger.info("SystemAPIClient initialized for direct interaction mode.")
            # In a real system, you'd initialize connection details here
            # self._api_base_url = "http://localhost:8000/api" # Example API base URL
        else:
            logger.warning("SystemAPIClient initialized in MOCK mode.")
            # In a real system, this would connect to a remote API/MQ
            # self._api_base_url = "http://localhost:8000/api" # Example API base URL


    def start_job(self, job_id):
        logger.info(f"Client requesting job start: {job_id}")
        if self._manager: # Interact directly with the real instance
            try:
                return self._manager.start_job_externally(job_id)
            except Exception as e:
                logger.error(f"Error calling real SystemManager start_job: {e}", exc_info=True)
                return False # Indicate failure
        else: # In MOCK mode
            logger.info(f"Mock Client: Requested job start: {job_id}")
            # In a real remote client, make API call/send MQ message
            # response = requests.post(f"{self._api_base_url}/jobs/start", json={'job_id': job_id})
            # return response.json().get('success', False) # Example response parsing
            return True # Simulate success in mock mode


    def stop_job(self):
        logger.info("Client requesting job stop.")
        if self._manager: # Interact directly with the real instance
             try:
                 return self._manager.stop_job_externally()
             except Exception as e:
                 logger.error(f"Error calling real SystemManager stop_job: {e}", exc_info=True)
                 return False # Indicate failure
        else: # In MOCK mode
            logger.info("Mock Client: Requested job stop.")
            # In a real remote client, make API call/send MQ message
            # response = requests.post(f"{self._api_base_url}/jobs/stop")
            # return response.json().get('success', False) # Example response parsing
            return True # Simulate success in mock mode


    def get_status(self):
        # logger.debug("Client requesting status.") # Too verbose
        if self._manager: # Interact directly with the real instance
             try:
                 status = self._manager.get_system_status_externally()
                 # Add a note if we are running the real manager in the same process for demo
                 status['note'] = "Interacting directly with REAL SystemManager instance."
                 return status
             except Exception as e:
                 logger.error(f"Error calling real SystemManager get_system_status: {e}", exc_info=True)
                 # If calling the real manager fails *after* it was initialized
                 return {
                    "manager_status": "REAL_MANAGER_ERROR",
                    "error": f"Error getting status from real manager: {e}",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "note": "Attempted to interact with real manager but failed."
                 }
        else: # In MOCK mode (SystemManager import or initialization failed)
            # In a real remote client, make API call
            # response = requests.get(f"{self._api_base_url}/status")
            # return response.json().get('status_data', {}) # Example response parsing

            # This is the mock status returned when SystemManager import/init failed
            return {
                "manager_status": "MOCK_FAILED_INIT", # Or just "MOCK_RUNNING" depending on desired mock state
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "note": "This is mock status data because SystemManager is not available.",
                "error": "SystemManager import or initialization failed." # Indicate why mock is used
            }

# --- Step 4: Initialize Flask App and API Client (will be done in __main__) ---
app = Flask(__name__)

# These will be initialized in the if __name__ == '__main__': block
system_manager_instance = None
api_client = None


# --- Flask Routes ---
@app.route('/')
def index():
    """Homepage."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Displays the current system status."""
    # The api_client is guaranteed to be initialized (either real or mock) in __main__
    system_status = api_client.get_status()
    return render_template('status.html', status=system_status)

@app.route('/jobs/start', methods=['POST'])
def start_job():
    """Handles request to start a welding job."""
    job_id = request.form.get('job_id')
    if not job_id:
        logger.error("Job ID not provided for start request.")
        # Ideally handle this on the frontend or provide a better error message
        return redirect(url_for('status')) # Redirect back to status with an error indicator

    logger.info(f"Received request to start job: {job_id}")
    success = api_client.start_job(job_id)

    if success:
        logger.info(f"Job '{job_id}' start request sent successfully via client.")
    else:
        logger.error(f"Failed to send job '{job_id}' start request via client.")

    # Redirect to status page to see updates
    return redirect(url_for('status'))

@app.route('/jobs/stop', methods=['POST'])
def stop_job():
    """Handles request to stop the current welding job."""
    logger.info("Received request to stop job.")
    success = api_client.stop_job()

    if success:
        logger.info("Job stop request sent successfully via client.")
    else:
        logger.error("Failed to send job stop request via client.")

    # Redirect to status page
    return redirect(url_for('status'))

# Optional: API endpoint for status (returns JSON)
@app.route('/api/status')
def api_status():
    """Returns system status as JSON."""
    system_status = api_client.get_status()
    return jsonify(system_status)


# --- Main execution block ---
if __name__ == '__main__':
    logger.info("--- Starting Flask HMI Application Process ---")

    # --- Step 5: Attempt to initialize SystemManager and the real API client ---
    if system_manager_available:
        try:
            # Initialize the SystemManager instance
            logger.info("Attempting to initialize SystemManager instance...")
            
            # Initialize core components first with config
            config = Config()
            data_logger = DataLoggerDB(config)
            robot_interface = RobotControlInterface(config)
            ai_engine = AIInferenceEngine(config)
            sensor_handler = SensorDataHandler(config)
            process_manager = WeldingProcessManager(config)
            
            # Initialize SystemManager with components
            system_manager_instance = SystemManager(
                config=config,
                data_logger=data_logger,
                robot_interface=robot_interface,
                ai_engine=ai_engine,
                sensor_handler=sensor_handler,
                process_manager=process_manager
            )
            
            logger.info("SystemManager instance created successfully.")

            # Initialize the API client with the SystemManager instance
            api_client = SystemAPIClient(manager_instance=system_manager_instance)
            logger.info("API Client initialized with REAL SystemManager instance.")

        except Exception as e:
            logger.critical(f"FATAL ERROR: Failed to initialize SystemManager instance: {e}", exc_info=True)
            logger.critical("HMI will interact with a MOCK client due to initialization failure.")
            api_client = SystemAPIClient(manager_instance=None)

    else:
        logger.critical("HMI will interact with a MOCK client because SystemManager import failed.")
        api_client = SystemAPIClient(manager_instance=None)

    # --- Step 6: Ensure api_client is initialized ---
    if api_client is None:
        logger.critical("API Client initialization logic failed unexpectedly. Creating a basic mock client.")
        class UltimateFallbackClient(SystemAPIClient):
            def get_status(self): return {"manager_status": "ULTIMATE_MOCK", "error": "API Client itself failed to initialize."}
        api_client = UltimateFallbackClient(manager_instance=None)

    # --- Step 7: Run the Flask Development Server ---
    logger.info("Running Flask server...")
    
    # Parse command line arguments for port
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='포트 번호 지정')
    args, unknown = parser.parse_known_args()

    # Determine port number with priority: 1) command line arg, 2) environment var, 3) default
    if args.port:
        port = args.port
    elif os.environ.get('PORT'):
        port = int(os.environ.get('PORT'))
    else:
        port = 5000

    app.run(debug=True, host='0.0.0.0', port=port)

    logger.info("--- Flask HMI Application Process Finished ---")
    
    # Step 8: Clean up on exit
    if system_manager_instance and hasattr(system_manager_instance, 'shutdown'):
        logger.info("Attempting to trigger SystemManager shutdown on Flask exit...")
        try:
            system_manager_instance.shutdown(graceful=True)
        except Exception as e:
            logger.error(f"Error during SystemManager shutdown attempt: {e}", exc_info=True)