# welding_process_manager.py
# Description: Manages the overall welding workflow, coordinating robots,
#              welding equipment (via robot IO), sensors, and AI modules.

import time
import logging
import threading
import enum # For defining process states or steps

import config # For general settings, perhaps welding process recipes
from robot_control_interface import RobotControlInterface # Assumed to be available
from quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl # For adaptive control
from ai_inference_engine import AIInferenceEngine # For real-time predictions
from sensor_data_handler import SensorDataHandler # For real-time sensor data (Placeholder)
# from data_logger_db import DataLoggerDB # For logging process data (Placeholder)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Process States (Example) ---
class WeldingProcessState(enum.Enum):
    IDLE = 0
    LOADING_PART = 1
    ROBOT_MOVING_TO_START = 2
    WELDING_IN_PROGRESS = 3
    ADAPTIVE_CONTROL_ACTIVE = 4 # Sub-state of WELDING_IN_PROGRESS
    WELDING_COMPLETED = 5
    ROBOT_MOVING_TO_HOME = 6
    UNLOADING_PART = 7
    ERROR = 8
    ABORTED = 9

# --- Welding Job/Recipe (Placeholder) ---
# In a real system, this would come from a database, UI, or config files.
# Example: A list of welding jobs, each with steps and parameters for each robot.
WELDING_JOBS_RECIPES = {
    "JOB_PART_A_SECTION_1": {
        "description": "Welding Section 1 of Part A",
        "robots_involved": [1], # List of robot IDs
        "steps": [
            {"robot_id": 1, "action": "move_to_start_pos", "target": "POS_PART_A_S1_START"}, # Virtual position name
            {"robot_id": 1, "action": "set_welding_params", "params": {"current": 150, "voltage": 22, "speed": 300}},
            {"robot_id": 1, "action": "welder_on"}, # Corresponds to robot IO
            {"robot_id": 1, "action": "run_job_program", "job_name": "WELD_PATH_A_S1"}, # Robot's internal program
            {"robot_id": 1, "action": "welder_off"},
            {"robot_id": 1, "action": "move_to_home_pos", "target": "POS_HOME_1"},
        ],
        "quality_checkpoints": [ # Times or events during "run_job_program" to get AI feedback
            {"time_elapsed_sec": 5, "check_id": "S1_CP1"},
            {"time_elapsed_sec": 15, "check_id": "S1_CP2"}
        ]
    },
    "JOB_PART_B_DUAL_ROBOT": {
        "description": "Welding Part B using two robots simultaneously",
        "robots_involved": [2, 3],
        "steps": [ # Steps can be defined for each robot or as coordinated steps
            {"robot_id": 2, "action": "move_to_start_pos", "target": "POS_PART_B_R2_START"},
            {"robot_id": 3, "action": "move_to_start_pos", "target": "POS_PART_B_R3_START"},
            # Coordinated start needs careful handling
            {"group_action": "set_welding_params_all", "robots": [2,3], "params": {"current": 160, "voltage": 23, "speed": 280}},
            {"group_action": "welder_on_all", "robots": [2,3]},
            {"group_action": "run_job_program_sync", # Requires robot controller support for sync
                "jobs": [{"robot_id": 2, "job_name": "WELD_PATH_B_R2"}, {"robot_id": 3, "job_name": "WELD_PATH_B_R3"}]
            },
            {"group_action": "welder_off_all", "robots": [2,3]},
            {"robot_id": 2, "action": "move_to_home_pos", "target": "POS_HOME_2"},
            {"robot_id": 3, "action": "move_to_home_pos", "target": "POS_HOME_3"},
        ],
         "quality_checkpoints": [
            {"job_event": "WELD_PATH_B_R2_MIDPOINT", "robot_id": 2, "check_id": "B_R2_CP1"}, # Event from robot
            {"job_event": "WELD_PATH_B_R3_MIDPOINT", "robot_id": 3, "check_id": "B_R3_CP1"}
        ]
    }
    # Add more job recipes...
}


class WeldingProcessManager:
    """
    Manages the entire welding workflow, coordinates equipment, and handles adaptive control.
    """
    def __init__(self, robot_interface: RobotControlInterface,
                 ai_engine: AIInferenceEngine,
                 quality_controller: QualityEvaluatorAdaptiveControl,
                 sensor_handler: SensorDataHandler # Placeholder
                 # data_logger: DataLoggerDB # Placeholder
                 ):
        self.robot_interface = robot_interface
        self.ai_engine = ai_engine
        self.quality_controller = quality_controller
        self.sensor_handler = sensor_handler # For getting real-time sensor data
        # self.data_logger = data_logger     # For logging everything

        self.current_job_id = None
        self.current_job_recipe = None
        self.current_process_state = WeldingProcessState.IDLE
        self.active_robots = {} # {robot_id: {"current_step_index": 0, "status": "idle", "current_params": {}}}
        self.process_thread = None
        self.stop_event = threading.Event() # To signal the process thread to stop

        logger.info("Welding Process Manager initialized.")

    def _set_process_state(self, state: WeldingProcessState):
        if self.current_process_state != state:
            logger.info(f"Process state changed from {self.current_process_state.name} to {state.name}")
            self.current_process_state = state
            # self.data_logger.log_process_state_change(self.current_job_id, state) # Example logging

    def _update_robot_status(self, robot_id, status_message, current_step_index=None):
        if robot_id in self.active_robots:
            self.active_robots[robot_id]["status"] = status_message
            if current_step_index is not None:
                self.active_robots[robot_id]["current_step_index"] = current_step_index
            # logger.debug(f"Robot {robot_id} status updated: {status_message}")

    def _get_current_robot_params(self, robot_id):
         return self.active_robots.get(robot_id, {}).get("current_params", {})

    def _update_robot_params(self, robot_id, new_params):
        if robot_id in self.active_robots:
            self.active_robots[robot_id].setdefault("current_params", {}).update(new_params)
            # logger.info(f"Robot {robot_id} parameters updated to: {self.active_robots[robot_id]['current_params']}")


    def _execute_robot_action(self, robot_id, action_details):
        """Executes a single action for a robot based on the recipe."""
        action_type = action_details.get("action")
        target = action_details.get("target")
        params = action_details.get("params")
        job_name = action_details.get("job_name")
        success = False

        logger.info(f"Robot {robot_id}: Executing action '{action_type}' with details: {action_details}")

        # Store current params if setting new ones
        if action_type == "set_welding_params" and params:
             self._update_robot_params(robot_id, params)

        if action_type == "move_to_start_pos" or action_type == "move_to_home_pos":
            # In a real system, "target" might be a pre-taught named position or coordinates.
            # The robot_interface would need a method like `move_to_named_position(robot_id, target)`
            # or `move_to_coordinates(robot_id, x, y, z, a, b, c)`.
            # Placeholder: Assume `run_job` can handle named positions for now.
            success = self.robot_interface.run_job(robot_id, job_name=target) # Virtual job for movement
        elif action_type == "set_welding_params":
            current_params = self._get_current_robot_params(robot_id)
            success = self.robot_interface.set_welding_parameters(robot_id,
                                                                current=current_params.get('current'),
                                                                voltage=current_params.get('voltage'),
                                                                speed=current_params.get('speed'))
        elif action_type == "welder_on":
            success = self.robot_interface.set_welder_io(robot_id, True) # True for ON
        elif action_type == "welder_off":
            success = self.robot_interface.set_welder_io(robot_id, False) # False for OFF
        elif action_type == "run_job_program":
            self._set_process_state(WeldingProcessState.WELDING_IN_PROGRESS)
            # This is where real-time monitoring and adaptive control would kick in
            # For simplicity, just run the job. A real system would monitor progress.
            success = self.robot_interface.run_job(robot_id, job_name=job_name)
            # After job completion (assuming it's synchronous for now), check quality if needed
            # or rely on periodic checkpoints.
        else:
            logger.warning(f"Robot {robot_id}: Unknown action type '{action_type}' in recipe.")
            success = False # Or treat as success if it's a comment/no-op

        if success:
            logger.info(f"Robot {robot_id}: Action '{action_type}' completed successfully.")
            self._update_robot_status(robot_id, f"Action '{action_type}' success")
        else:
            logger.error(f"Robot {robot_id}: Action '{action_type}' failed.")
            self._update_robot_status(robot_id, f"Action '{action_type}' FAILED")
            self._set_process_state(WeldingProcessState.ERROR)
        return success

    def _execute_group_action(self, group_action_details):
        """Executes an action involving multiple robots."""
        action_type = group_action_details.get("group_action")
        robot_ids = group_action_details.get("robots", [])
        params = group_action_details.get("params")
        jobs_info = group_action_details.get("jobs") # List of {"robot_id": id, "job_name": name}
        all_success = True

        logger.info(f"Executing group action '{action_type}' for robots {robot_ids} with details: {group_action_details}")

        if action_type == "set_welding_params_all":
            for robot_id in robot_ids:
                 self._update_robot_params(robot_id, params) # Store for each robot
                 if not self.robot_interface.set_welding_parameters(robot_id,
                                                                    current=params.get('current'),
                                                                    voltage=params.get('voltage'),
                                                                    speed=params.get('speed')):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to set welding params in group action.")
                    break # Or continue and report partial failure
        elif action_type == "welder_on_all":
            for robot_id in robot_ids:
                if not self.robot_interface.set_welder_io(robot_id, True):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to turn welder on in group action.")
                    break
        elif action_type == "welder_off_all":
            for robot_id in robot_ids:
                if not self.robot_interface.set_welder_io(robot_id, False):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to turn welder off in group action.")
                    break
        elif action_type == "run_job_program_sync":
            # This requires the robot_interface and robot controllers to support
            # synchronized start of jobs. For this placeholder, we run them sequentially
            # or assume the interface handles true synchronization.
            logger.warning("Executing 'run_job_program_sync' - True synchronization depends on robot controller/interface capabilities.")
            self._set_process_state(WeldingProcessState.WELDING_IN_PROGRESS)
            threads = []
            results_queue = queue.Queue()

            def _run_single_job_in_group(r_id, j_name):
                 if self.robot_interface.run_job(r_id, job_name=j_name):
                     results_queue.put({r_id: True})
                 else:
                     results_queue.put({r_id: False})

            for job_info in jobs_info:
                r_id = job_info["robot_id"]
                j_name = job_info["job_name"]
                thread = threading.Thread(target=_run_single_job_in_group, args=(r_id, j_name))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join() # Wait for all jobs to complete

            # Check results
            while not results_queue.empty():
                result = results_queue.get()
                for r_id, success_status in result.items():
                     if not success_status:
                         all_success = False
                         logger.error(f"Robot {r_id}: Job failed within synchronized group.")
                         self._update_robot_status(r_id, f"Job FAILED in sync group")
                     else:
                         self._update_robot_status(r_id, f"Job success in sync group")

        else:
            logger.warning(f"Unknown group action type '{action_type}'.")
            all_success = False

        if all_success:
            logger.info(f"Group action '{action_type}' completed successfully.")
        else:
            logger.error(f"Group action '{action_type}' failed for one or more robots.")
            self._set_process_state(WeldingProcessState.ERROR)
        return all_success


    def _adaptive_control_loop(self, robot_id, welding_job_name, checkpoints):
        """
        Monitors welding and applies adaptive control during a specific robot's job.
        This should run in a separate thread per actively welding robot.
        """
        logger.info(f"Robot {robot_id}: Starting adaptive control loop for job '{welding_job_name}'.")
        self._set_process_state(WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE)
        job_start_time = time.time()
        checkpoint_idx = 0

        while self.current_process_state == WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE and \
              not self.stop_event.is_set():
            # Check for job completion (this is tricky, needs robot feedback)
            # For now, assume job completion is handled outside this loop or by timeout/event.

            # 1. Get Real-time Sensor Data (from sensor_handler)
            # This is a placeholder. sensor_handler needs to provide this.
            current_sensor_data = self.sensor_handler.get_latest_aggregated_data(robot_id)
            if not current_sensor_data:
                logger.warning(f"Robot {robot_id}: No sensor data available for adaptive control. Skipping cycle.")
                time.sleep(config.ADAPTIVE_CONTROL_CYCLE_TIME_SEC) # Wait before next check
                continue

            # 2. Get AI Prediction (from ai_engine)
            ai_prediction = self.ai_engine.process_sensor_data(current_sensor_data['values_for_ai']) # Assuming sensor_handler prepares this
            # logger.debug(f"Robot {robot_id}: AI Prediction for AC: {ai_prediction}")


            # 3. Evaluate Quality (from quality_controller)
            quality_evaluation = self.quality_controller.evaluate_quality(
                ai_prediction=ai_prediction,
                real_time_sensor_data=current_sensor_data # Pass full sensor dict
            )
            # logger.debug(f"Robot {robot_id}: Quality Eval for AC: {quality_evaluation}")


            # 4. Generate Adaptive Adjustments (from quality_controller)
            current_robot_params = self._get_current_robot_params(robot_id)
            adjustments = self.quality_controller.generate_adaptive_adjustments(
                evaluation_results=quality_evaluation,
                current_welding_params=current_robot_params
            )

            # 5. Apply Adjustments (via robot_interface)
            if adjustments:
                logger.info(f"Robot {robot_id}: Applying adaptive adjustments: {adjustments}")
                # Update stored current params first
                self._update_robot_params(robot_id, adjustments) # Store new target params
                # Send to robot (this is simplified, might need specific methods)
                if not self.robot_interface.set_welding_parameters(robot_id,
                                                                  current=adjustments.get('current', current_robot_params.get('current')),
                                                                  voltage=adjustments.get('voltage', current_robot_params.get('voltage')),
                                                                  speed=adjustments.get('speed', current_robot_params.get('speed'))):
                    logger.error(f"Robot {robot_id}: Failed to apply adaptive adjustments via robot interface.")
                    # Potentially revert stored params or enter error state
                else:
                    logger.info(f"Robot {robot_id}: Adaptive adjustments applied successfully.")
                # Add logic for adjusting other params like torch_angle if they are in 'adjustments'
            else:
                # logger.debug(f"Robot {robot_id}: No adaptive adjustments needed this cycle.")
                pass

            # Handle recipe-defined quality checkpoints (simplified)
            time_elapsed = time.time() - job_start_time
            if checkpoints and checkpoint_idx < len(checkpoints):
                checkpoint = checkpoints[checkpoint_idx]
                if "time_elapsed_sec" in checkpoint and time_elapsed >= checkpoint["time_elapsed_sec"]:
                    logger.info(f"Robot {robot_id}: Reached quality checkpoint {checkpoint['check_id']} at {time_elapsed:.2f}s.")
                    # Log quality_evaluation and ai_prediction for this checkpoint
                    # self.data_logger.log_quality_checkpoint(...)
                    checkpoint_idx += 1
                # Add logic for "job_event" based checkpoints if robot can send such events

            # Control loop frequency
            time.sleep(getattr(config, 'ADAPTIVE_CONTROL_CYCLE_TIME_SEC', 1.0)) # Default 1 sec

        logger.info(f"Robot {robot_id}: Adaptive control loop for job '{welding_job_name}' ended.")
        if self.current_process_state == WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE:
             # If loop ended but state is still AC, it means job probably finished
             # or was aborted. Reset to WELDING_IN_PROGRESS or IDLE based on broader context.
             self._set_process_state(WeldingProcessState.WELDING_IN_PROGRESS) # Or IDLE

    def _process_job_thread_func(self, job_id):
        """The main logic for processing a single welding job."""
        self.current_job_id = job_id
        self.current_job_recipe = WELDING_JOBS_RECIPES.get(job_id)

        if not self.current_job_recipe:
            logger.error(f"Job ID '{job_id}' not found in recipes. Aborting.")
            self._set_process_state(WeldingProcessState.ERROR)
            return

        logger.info(f"Starting to process welding job: {job_id} - {self.current_job_recipe['description']}")
        self._set_process_state(WeldingProcessState.LOADING_PART) # Example initial state
        self.stop_event.clear() # Ensure stop event is cleared at start

        # Initialize active robots for this job
        self.active_robots = {
            robot_id: {"current_step_index": 0, "status": "pending", "current_params": {}}
            for robot_id in self.current_job_recipe.get("robots_involved", [])
        }
        # If group actions, ensure all robots in group are in active_robots
        for step in self.current_job_recipe.get("steps", []):
             if "group_action" in step:
                 for r_id in step.get("robots", []):
                     if r_id not in self.active_robots:
                         self.active_robots[r_id] = {"current_step_index": 0, "status": "pending", "current_params": {}}


        # --- Connect to necessary robots ---
        # For simplicity, assume robot_interface.connect_all() was called before starting job.
        # A more robust system might connect only to robots_involved if not already connected.
        if not self.robot_interface.connected_robot_ids:
             logger.error("No robots connected. Cannot start job.")
             self._set_process_state(WeldingProcessState.ERROR)
             return

        # --- Execute Recipe Steps ---
        all_steps = self.current_job_recipe.get("steps", [])
        current_step_num = 0
        adaptive_control_threads = {} # To manage AC loops

        while current_step_num < len(all_steps) and \
              self.current_process_state not in [WeldingProcessState.ERROR, WeldingProcessState.ABORTED] and \
              not self.stop_event.is_set():

            step_details = all_steps[current_step_num]
            logger.info(f"Job '{job_id}': Executing Step {current_step_num + 1}/{len(all_steps)}")

            success = False
            if "action" in step_details: # Single robot action
                robot_id = step_details.get("robot_id")
                if robot_id not in self.robot_interface.connected_robot_ids:
                    logger.error(f"Robot {robot_id} for step {current_step_num+1} is not connected. Aborting step.")
                    self._set_process_state(WeldingProcessState.ERROR)
                    break
                self._update_robot_status(robot_id, f"Executing step {current_step_num+1}", current_step_num)
                success = self._execute_robot_action(robot_id, step_details)

                # If this step was 'run_job_program', start adaptive control for this robot
                if success and step_details.get("action") == "run_job_program":
                    job_name = step_details.get("job_name")
                    checkpoints = self.current_job_recipe.get("quality_checkpoints", [])
                    # Filter checkpoints relevant to this robot's job if defined per robot
                    robot_checkpoints = [cp for cp in checkpoints if cp.get("robot_id") == robot_id or "robot_id" not in cp]

                    ac_thread = threading.Thread(target=self._adaptive_control_loop,
                                                 args=(robot_id, job_name, robot_checkpoints), daemon=True)
                    adaptive_control_threads[robot_id] = ac_thread
                    ac_thread.start()

            elif "group_action" in step_details: # Group robot action
                 success = self._execute_group_action(step_details)
                 # Start AC for all robots in the group if it was a synchronized run_job_program
                 if success and step_details.get("group_action") == "run_job_program_sync":
                     jobs_info = step_details.get("jobs", [])
                     checkpoints = self.current_job_recipe.get("quality_checkpoints", [])
                     for job_info in jobs_info:
                         r_id = job_info["robot_id"]
                         j_name = job_info["job_name"]
                         robot_checkpoints = [cp for cp in checkpoints if cp.get("robot_id") == r_id or "robot_id" not in cp]
                         if r_id not in adaptive_control_threads or not adaptive_control_threads[r_id].is_alive():
                             ac_thread = threading.Thread(target=self._adaptive_control_loop,
                                                          args=(r_id, j_name, robot_checkpoints), daemon=True)
                             adaptive_control_threads[r_id] = ac_thread
                             ac_thread.start()
            else:
                logger.warning(f"Job '{job_id}': Step {current_step_num + 1} has no recognized action. Skipping.")
                success = True # Treat as success to proceed

            if not success:
                logger.error(f"Job '{job_id}': Failed at step {current_step_num + 1}. Process halting.")
                self._set_process_state(WeldingProcessState.ERROR)
                break # Exit step loop on failure

            current_step_num += 1
            # Add delay between steps if needed
            # time.sleep(config.INTER_STEP_DELAY_SEC)

        # --- Wait for all adaptive control loops to finish (if any started) ---
        # This assumes AC loops will self-terminate when their welding job is done or if state changes
        # Or the main process state change (ERROR/ABORTED) signals them to stop
        if self.current_process_state not in [WeldingProcessState.ERROR, WeldingProcessState.ABORTED]:
             self._set_process_state(WeldingProcessState.WELDING_COMPLETED) # If all steps done

        for robot_id, thread in adaptive_control_threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for adaptive control for Robot {robot_id} to finish...")
                # Signal AC loop to stop if not already due to state change
                # (This is tricky, AC loop needs robust termination logic based on robot job status)
                if self.current_process_state != WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE:
                    # The AC loop should notice the state change and exit.
                    # Forcing stop_event can be an additional measure.
                    # self.stop_event.set() # This would stop ALL AC loops if they check it.
                    pass
                thread.join(timeout=30) # Max wait for AC loop
                if thread.is_alive():
                     logger.warning(f"Adaptive control thread for Robot {robot_id} did not terminate.")

        # --- Finalize Job ---
        if self.stop_event.is_set() and self.current_process_state != WeldingProcessState.ERROR:
             self._set_process_state(WeldingProcessState.ABORTED)
             logger.info(f"Job '{job_id}' was aborted by external signal.")
        elif self.current_process_state == WeldingProcessState.WELDING_COMPLETED:
            logger.info(f"Job '{job_id}' completed successfully.")
            # Perform any post-job actions, e.g., final quality report generation
        elif self.current_process_state == WeldingProcessState.ERROR:
            logger.error(f"Job '{job_id}' ended with an error.")
            # Perform error recovery or notification

        self._set_process_state(WeldingProcessState.IDLE) # Reset manager to idle
        self.current_job_id = None
        self.current_job_recipe = None
        self.active_robots = {}
        logger.info(f"Welding Process Manager returned to IDLE state.")

    def start_welding_job(self, job_id):
        """Starts a new welding job in a separate thread."""
        if self.current_process_state != WeldingProcessState.IDLE:
            logger.error(f"Cannot start job '{job_id}'. Manager is not IDLE (current state: {self.current_process_state.name}).")
            return False

        if not WELDING_JOBS_RECIPES.get(job_id):
             logger.error(f"Job ID '{job_id}' not found in recipes.")
             return False

        logger.info(f"Queuing welding job: {job_id}")
        self.process_thread = threading.Thread(target=self._process_job_thread_func, args=(job_id,), daemon=True)
        self.process_thread.start()
        return True

    def stop_current_job(self):
        """Requests to stop the currently running job."""
        if self.current_process_state != WeldingProcessState.IDLE and self.process_thread and self.process_thread.is_alive():
            logger.warning(f"Attempting to stop current job: {self.current_job_id}")
            self.stop_event.set() # Signal the process thread and AC loops to stop
            # Also send stop commands to all active robots
            for robot_id in list(self.active_robots.keys()): # Make a copy
                logger.info(f"Sending STOP command to Robot {robot_id}")
                self.robot_interface.stop_job(robot_id) # This should be non-blocking or quick
            # The process thread should detect stop_event and transition to ABORTED/IDLE
            return True
        else:
            logger.info("No job currently running or manager is IDLE.")
            return False

    def get_manager_status(self):
        """Returns the current status of the process manager."""
        return {
            "current_job_id": self.current_job_id,
            "process_state": self.current_process_state.name,
            "active_robots": self.active_robots,
            "stop_event_set": self.stop_event.is_set()
        }

# Example Usage (requires other modules to be instantiated)
if __name__ == '__main__':
    logger.info("--- Welding Process Manager Example ---")

    # --- Dummy Config and Module Instantiation (Highly Simplified) ---
    class DummyConfig:
        ADAPTIVE_CONTROL_CYCLE_TIME_SEC = 0.5 # Faster for example
        ROBOT_CONFIGS = [{'id': 1, 'ip': '127.0.0.1', 'port': 6001}] # For robot_interface
        # ... other configs needed by imported modules
    config = DummyConfig()

    # Dummy Robot Server Thread (from robot_control_interface example)
    # (Copy the dummy_robot_server function here or import it if modularized)
    def dummy_robot_server_wpm(robot_id, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((host, port))
            server_socket.listen(1)
            logger.info(f"[DummyRobotWPM {robot_id}] Listening on {host}:{port}")
            client_socket, addr = server_socket.accept() # Accept only one connection for simplicity
            logger.info(f"[DummyRobotWPM {robot_id}] Accepted connection from {addr}")
            with client_socket:
                while True:
                    raw_msglen = client_socket.recv(4)
                    if not raw_msglen: break
                    msglen = int.from_bytes(raw_msglen, 'big')
                    data_bytes = client_socket.recv(msglen)
                    if not data_bytes: break
                    request = json.loads(data_bytes.decode('utf-8'))
                    logger.debug(f"[DummyRobotWPM {robot_id}] Received: {request}")
                    response = {"status": "success", "sequence_id": request["sequence_id"], "data": {}}
                    if request['action'] == "run_job": # Simulate job taking time
                         logger.info(f"[DummyRobotWPM {robot_id}] Simulating job '{request['parameters']['job_name']}' for 3s...")
                         time.sleep(3)
                         response['message'] = f"Job {request['parameters']['job_name']} completed."
                    client_socket.sendall(len(json.dumps(response).encode('utf-8')).to_bytes(4, 'big'))
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e: logger.error(f"[DummyRobotWPM {robot_id}] Server error: {e}")
        finally: server_socket.close()

    # Start dummy robot server
    server_thread = threading.Thread(target=dummy_robot_server_wpm, args=(1, '127.0.0.1', 6001), daemon=True)
    server_thread.start()
    time.sleep(0.2) # Give server a moment

    # Instantiate dependencies (simplified versions)
    class DummyAIInferenceEngine:
        def process_sensor_data(self, data): return {'predicted_class': 0, 'confidence': 0.95}
    class DummyQualityEvaluatorAdaptiveControl:
        def evaluate_quality(self, ai_prediction, real_time_sensor_data=None, current_welding_params=None):
            return {'combined_status': 'Good', 'detected_issues': []}
        def generate_adaptive_adjustments(self, evaluation_results, current_welding_params):
            if evaluation_results['combined_status'] != 'Good': return {'speed': -5} # Example adjustment
            return None
    class DummySensorDataHandler:
        def get_latest_aggregated_data(self, robot_id): return {'values_for_ai': [0.1, 0.2, 0.3, 0.4], 'temperature': 300} # Dummy data

    mock_robot_interface = RobotControlInterface() # Assumes config has ROBOT_CONFIGS
    mock_ai_engine = DummyAIInferenceEngine()
    mock_quality_controller = DummyQualityEvaluatorAdaptiveControl()
    mock_sensor_handler = DummySensorDataHandler()

    # --- Initialize Process Manager ---
    process_manager = WeldingProcessManager(
        robot_interface=mock_robot_interface,
        ai_engine=mock_ai_engine,
        quality_controller=mock_quality_controller,
        sensor_handler=mock_sensor_handler
    )

    # Connect to robots
    if not mock_robot_interface.connect_all():
        logger.error("Failed to connect to dummy robot for example. Exiting.")
    else:
        # --- Start a Welding Job ---
        job_to_run = "JOB_PART_A_SECTION_1"
        if process_manager.start_welding_job(job_to_run):
            logger.info(f"Job '{job_to_run}' started. Monitoring manager status...")
            for _ in range(10): # Monitor for a few seconds
                status = process_manager.get_manager_status()
                logger.info(f"Manager Status: {status}")
                if status["process_state"] == "IDLE" and status["current_job_id"] is None:
                    logger.info("Job processing finished.")
                    break
                time.sleep(1)

            # Example: Stop job if still running (for testing stop_current_job)
            if process_manager.current_process_state != WeldingProcessState.IDLE:
                logger.info("Attempting to stop job prematurely...")
                process_manager.stop_current_job()
                time.sleep(2) # Give time for stop to process
                logger.info(f"Final Manager Status after stop: {process_manager.get_manager_status()}")
        else:
            logger.error(f"Failed to start job '{job_to_run}'.")

        mock_robot_interface.disconnect_all()

    logger.info("--- Welding Process Manager Example Finished ---")