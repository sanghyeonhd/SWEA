# src/welding_process_manager.py
# Description: Manages the overall welding workflow, coordinating robots,
#              welding equipment (via robot IO), sensors, and AI modules.
#              Also sends visualization commands to Unreal Engine via physics_interface.

import time
import logging
import threading
import enum # For defining process states or steps
import queue # For managing internal queues or data passing

import config # For general settings, perhaps welding process recipes
# Import necessary modules (SystemManager will inject instances)
from robot_control_interface import RobotControlInterface
from quality_evaluator_adaptive_control import QualityEvaluatorAdaptiveControl
from ai_inference_engine import AIInferenceEngine
from sensor_data_handler import SensorDataHandler
from data_logger_db import DataLoggerDB
from physics_interface import UnrealSimulatorInterface # Import the physics interface

# Setup logging for this module
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
# ADDED: 'target_pose' field in move actions for visualization reference in UE.
WELDING_JOBS_RECIPES = {
    "JOB_PART_A_SECTION_1": {
        "description": "Welding Section 1 of Part A",
        "robots_involved": [1], # List of robot IDs
        "steps": [
            {"robot_id": 1, "action": "move_to_start_pos", "target": "POS_PART_A_S1_START",
             "target_pose": {'joint_angles': [10, -5, 90, 0, 5, -10], 'tcp_transform': {'position': [500, 100, 200], 'rotation': [0.707, 0, 0, 0.707]}}}, # <-- ADDED target_pose
            {"robot_id": 1, "action": "set_welding_params", "params": {"current": 150, "voltage": 22, "speed": 300}},
            {"robot_id": 1, "action": "welder_on"},
            {"robot_id": 1, "action": "run_job_program", "job_name": "WELD_PATH_A_S1"}, # Robot's internal program
            {"robot_id": 1, "action": "welder_off"},
            {"robot_id": 1, "action": "move_to_home_pos", "target": "POS_HOME_1",
             "target_pose": {'joint_angles': [0, 0, 0, 0, 0, 0], 'tcp_transform': {'position': [0, 0, 500], 'rotation': [1, 0, 0, 0]}}}, # <-- ADDED target_pose
        ],
        "quality_checkpoints": [ # Times or events during "run_job_program" to get AI feedback
            {"time_elapsed_sec": 5, "check_id": "S1_CP1", "robot_id": 1}, # Added robot_id for clarity
            {"time_elapsed_sec": 15, "check_id": "S1_CP2", "robot_id": 1}
        ]
    },
    "JOB_PART_B_DUAL_ROBOT": {
        "description": "Welding Part B using two robots simultaneously",
        "robots_involved": [2, 3],
        "steps": [ # Steps can be defined for each robot or as coordinated steps
            {"robot_id": 2, "action": "move_to_start_pos", "target": "POS_PART_B_R2_START",
             "target_pose": {'joint_angles': [15, 10, 80, 0, 10, -5], 'tcp_transform': {'position': [600, -100, 250], 'rotation': [0.707, 0, 0, 0.707]}}},
            {"robot_id": 3, "action": "move_to_start_pos", "target": "POS_PART_B_R3_START",
             "target_pose": {'joint_angles': [15, -10, 80, 0, -10, -5], 'tcp_transform': {'position': [600, 100, 250], 'rotation': [0.707, 0, 0, 0.707]}}},
            # Coordinated start needs careful handling
            {"group_action": "set_welding_params_all", "robots": [2,3], "params": {"current": 160, "voltage": 23, "speed": 280}},
            {"group_action": "welder_on_all", "robots": [2,3]},
            {"group_action": "run_job_program_sync", # Requires robot controller support for sync
                "jobs": [{"robot_id": 2, "job_name": "WELD_PATH_B_R2"}, {"robot_id": 3, "job_name": "WELD_PATH_B_R3"}]
            },
            {"group_action": "welder_off_all", "robots": [2,3]},
            {"robot_id": 2, "action": "move_to_home_pos", "target": "POS_HOME_2",
             "target_pose": {'joint_angles': [0, 0, 0, 0, 0, 0], 'tcp_transform': {'position': [0, 0, 500], 'rotation': [1, 0, 0, 0]}}},
            {"robot_id": 3, "action": "move_to_home_pos", "target": "POS_HOME_3",
             "target_pose": {'joint_angles': [0, 0, 0, 0, 0, 0], 'tcp_transform': {'position': [0, 0, 500], 'rotation': [1, 0, 0, 0]}}},
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
    Communicates with Unreal Engine for visualization updates.
    """
    def __init__(self, robot_interface: RobotControlInterface,
                 ai_engine: AIInferenceEngine,
                 quality_controller: QualityEvaluatorAdaptiveControl,
                 sensor_handler: SensorDataHandler, # Placeholder
                 data_logger: DataLoggerDB, # Placeholder
                 physics_interface: UnrealSimulatorInterface # <-- Receive physics_interface instance
                 ):
        self.robot_interface = robot_interface
        self.ai_engine = ai_engine
        self.quality_controller = quality_controller
        self.sensor_handler = sensor_handler # For getting real-time sensor data
        self.data_logger = data_logger     # For logging everything
        self.physics_interface = physics_interface # <-- Store physics_interface instance

        self.current_job_id = None
        self.current_job_recipe = None
        self.current_process_state = WeldingProcessState.IDLE
        self.active_robots = {} # {robot_id: {"current_step_index": 0, "status": "idle", "current_params": {}}}
        self.process_thread = None
        self.stop_event = threading.Event() # To signal the process thread to stop
        self._adaptive_control_threads = {} # To manage AC loops per robot

        logger.info("Welding Process Manager initialized.")

    def _set_process_state(self, state: WeldingProcessState):
        if self.current_process_state != state:
            logger.info(f"Process state changed from {self.current_process_state.name} to {state.name}")
            self.current_process_state = state
            # Log state change event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="PROCESS_STATE_CHANGE",
                     job_id=self.current_job_id,
                     details={"old_state": self.current_process_state.name, "new_state": state.name}
                 )


    def _update_robot_status(self, robot_id, status_message, current_step_index=None):
        if robot_id in self.active_robots:
            self.active_robots[robot_id]["status"] = status_message
            if current_step_index is not None:
                self.active_robots[robot_id]["current_step_index"] = current_step_index
            # logger.debug(f"Robot {robot_id} status updated: {status_message}")
            # Consider logging robot status updates if they are significant
            # if self.data_logger:
            #      self.data_logger.log_robot_status(robot_id, {"status_message": status_message}, job_id=self.current_job_id)


    def _get_current_robot_params(self, robot_id):
         return self.active_robots.get(robot_id, {}).get("current_params", {})

    def _update_robot_params(self, robot_id, new_params):
        if robot_id in self.active_robots:
            # Ensure current_params dict exists
            if "current_params" not in self.active_robots[robot_id]:
                 self.active_robots[robot_id]["current_params"] = {}
            self.active_robots[robot_id]["current_params"].update(new_params)
            logger.info(f"Robot {robot_id} internal parameters updated: {self.active_robots[robot_id]['current_params']}")
            # Log parameter update event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="PARAMETER_UPDATE",
                     job_id=self.current_job_id,
                     robot_id=robot_id,
                     details={"updated_params": new_params, "all_params": self.active_robots[robot_id]["current_params"]}
                 )


    def _execute_robot_action(self, robot_id, action_details):
        """Executes a single action for a robot based on the recipe."""
        action_type = action_details.get("action")
        target = action_details.get("target")
        params = action_details.get("params")
        job_name = action_details.get("job_name")
        target_pose = action_details.get("target_pose") # <-- Get target_pose for visualization
        success = False

        logger.info(f"Robot {robot_id}: Executing action '{action_type}' with details: {action_details}")

        # Log action execution event
        if self.data_logger:
             self.data_logger.log_process_event(
                 event_type="ACTION_EXECUTION",
                 job_id=self.current_job_id,
                 robot_id=robot_id,
                 details={"action_type": action_type, "details": action_details}
             )


        # Store current params if setting new ones
        if action_type == "set_welding_params" and params:
             self._update_robot_params(robot_id, params)
             # No direct UE visual update needed for just setting internal params


        # --- Actions that involve sending commands to the real robot/sim via RobotControlInterface ---
        # These often need a corresponding visualization update in UE.
        if action_type == "move_to_start_pos" or action_type == "move_to_home_pos":
            # Send the move command to the robot (or motion simulation).
            # Assuming run_job with target name is sufficient for robot interface placeholder.
            # In a real system, this might be a specific `move_to_pos` command.
            success = self.robot_interface.run_job(robot_id, job_name=target) # Use target as job name for placeholder

            # --- Notify UE of the target pose for visualization ---
            # We send the *target* pose immediately, assuming the visualization will
            # animate towards it or jump there. For smooth animation,
            # a real-time pose stream during the move is needed (handled below in AC loop conceptually).
            if success and target_pose and self.physics_interface and self.physics_interface.is_connected:
                 logger.debug(f"Robot {robot_id}: Notifying UE of target pose for '{action_type}': {target}")
                 self.physics_interface.send_robot_pose(
                     robot_id=robot_id,
                     joint_angles=target_pose.get('joint_angles'),
                     tcp_transform=target_pose.get('tcp_transform')
                 )
            elif target_pose and not self.physics_interface:
                 logger.warning(f"Robot {robot_id}: Cannot send target pose to UE. physics_interface not available.")


        elif action_type == "welder_on":
            # Send command to turn on the welder (via robot IO)
            success = self.robot_interface.set_welder_io(robot_id, True) # True for ON

            # --- Notify UE to visualize arc ON ---
            if success and self.physics_interface and self.physics_interface.is_connected:
                 logger.info(f"Robot {robot_id}: Notifying UE to turn arc ON.")
                 # You might send current welding parameters here for UE visual fidelity
                 current_params = self._get_current_robot_params(robot_id)
                 self.physics_interface.send_welding_visual_command(
                     robot_id=robot_id,
                     command_type="arc_on",
                     details={"welding_params": current_params} # Send params for visual tuning
                 )
            elif self.physics_interface and not self.physics_interface.is_connected:
                 logger.warning(f"Robot {robot_id}: Cannot send welder ON command to UE. physics_interface not connected.")


        elif action_type == "welder_off":
            # Send command to turn off the welder
            success = self.robot_interface.set_welder_io(robot_id, False) # False for OFF

            # --- Notify UE to visualize arc OFF ---
            if success and self.physics_interface and self.physics_interface.is_connected:
                 logger.info(f"Robot {robot_id}: Notifying UE to turn arc OFF.")
                 self.physics_interface.send_welding_visual_command(
                     robot_id=robot_id,
                     command_type="arc_off"
                 )
            elif self.physics_interface and not self.physics_interface.is_connected:
                 logger.warning(f"Robot {robot_id}: Cannot send welder OFF command to UE. physics_interface not connected.")


        elif action_type == "run_job_program":
            self._set_process_state(WeldingProcessState.WELDING_IN_PROGRESS)
            # Send command to run the robot's internal welding job program
            success = self.robot_interface.run_job(robot_id, job_name=job_name)
            # In a real system, after sending the command, you need to monitor robot status
            # to know when the job actually starts and ends.
            # This is also the point where real-time pose streaming to UE should begin.
            # The _adaptive_control_loop for this robot starts *after* this command is sent successfully.
            # The AC loop is responsible for getting real-time pose updates and sending them to UE.

        else:
            logger.warning(f"Robot {robot_id}: Unknown action type '{action_type}' in recipe.")
            success = False # Or treat as success if it's a comment/no-op


        # --- Log Action Outcome ---
        if success:
            logger.info(f"Robot {robot_id}: Action '{action_type}' command sent successfully.")
            self._update_robot_status(robot_id, f"Sent action '{action_type}'")
            # Note: Command sent successfully doesn't mean action completed successfully.
            # Real status updates need to be monitored.
        else:
            logger.error(f"Robot {robot_id}: Action '{action_type}' command failed to send.")
            self._update_robot_status(robot_id, f"Action '{action_type}' FAILED TO SEND")
            self._set_process_state(WeldingProcessState.ERROR) # Fatal error if command can't be sent
        return success


    def _execute_group_action(self, group_action_details):
        """Executes an action involving multiple robots."""
        action_type = group_action_details.get("group_action")
        robot_ids = group_action_details.get("robots", [])
        params = group_action_details.get("params")
        jobs_info = group_action_details.get("jobs") # List of {"robot_id": id, "job_name": name}
        all_success = True

        logger.info(f"Executing group action '{action_type}' for robots {robot_ids} with details: {group_action_details}")

        # Log group action execution event
        if self.data_logger:
             self.data_logger.log_process_event(
                 event_type="GROUP_ACTION_EXECUTION",
                 job_id=self.current_job_id,
                 details={"action_type": action_type, "robots": robot_ids, "details": group_action_details}
             )


        # --- Actions that involve sending commands to multiple robots ---
        if action_type == "set_welding_params_all":
            for robot_id in robot_ids:
                 self._update_robot_params(robot_id, params) # Store for each robot
                 if not self.robot_interface.set_welding_parameters(robot_id,
                                                                    current=params.get('current'),
                                                                    voltage=params.get('voltage'),
                                                                    speed=params.get('speed')):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to set welding params in group action.")
                    # Break or continue depends on whether partial success is acceptable
                    # For robustness, might revert already set params on failure
                    break
        elif action_type == "welder_on_all":
            for robot_id in robot_ids:
                if not self.robot_interface.set_welder_io(robot_id, True):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to turn welder on in group action.")
                    break
                # Notify UE for each robot in the group
                if self.physics_interface and self.physics_interface.is_connected:
                     logger.debug(f"Robot {robot_id}: Notifying UE (group) to turn arc ON.")
                     current_params = self._get_current_robot_params(robot_id)
                     self.physics_interface.send_welding_visual_command(
                         robot_id=robot_id, command_type="arc_on", details={"welding_params": current_params}
                     )
        elif action_type == "welder_off_all":
            for robot_id in robot_ids:
                if not self.robot_interface.set_welder_io(robot_id, False):
                    all_success = False
                    logger.error(f"Robot {robot_id}: Failed to turn welder off in group action.")
                    break
                 # Notify UE for each robot in the group
                if self.physics_interface and self.physics_interface.is_connected:
                     logger.debug(f"Robot {robot_id}: Notifying UE (group) to turn arc OFF.")
                     self.physics_interface.send_welding_visual_command(
                         robot_id=robot_id, command_type="arc_off"
                     )
        elif action_type == "run_job_program_sync":
            # This requires synchronized start on the robot controller side.
            # We'll send the commands, and start AC loops.
            logger.warning("Executing 'run_job_program_sync' - True synchronization depends on robot controller/interface capabilities.")
            self._set_process_state(WeldingProcessState.WELDING_IN_PROGRESS)
            threads = []
            results_queue = queue.Queue() # Queue to collect results from job threads

            def _run_single_job_in_group(r_id, j_name):
                 # This thread only *sends* the run job command and waits for its acknowledgment.
                 # It does NOT wait for the job itself to complete on the robot.
                 # Job completion monitoring is handled by robot_control_interface status streaming
                 # or by polling in a different part of the WPM or AC loop.
                 if self.robot_interface.run_job(r_id, job_name=j_name):
                     results_queue.put({r_id: True})
                 else:
                     results_queue.put({r_id: False})

            # Start threads to send run job commands to each robot in the group
            for job_info in jobs_info:
                r_id = job_info["robot_id"]
                j_name = job_info["job_name"]
                thread = threading.Thread(target=_run_single_job_in_group, args=(r_id, j_name))
                threads.append(thread)
                thread.start()

            # Wait for all "run_job" commands to be acknowledged by the robot interface
            # This does NOT mean the jobs are running in sync on the robots yet.
            for thread in threads:
                thread.join()

            # Collect results and check if all commands were sent successfully
            sent_success = True
            failed_robots = []
            while not results_queue.empty():
                result = results_queue.get()
                for r_id, success_status in result.items():
                     if not success_status:
                         sent_success = False
                         failed_robots.append(r_id)
                         logger.error(f"Robot {r_id}: Failed to send run job command in sync group.")
                         self._update_robot_status(r_id, f"Run job command FAILED")
                     else:
                          self._update_robot_status(r_id, f"Run job command SENT")

            if sent_success:
                # If commands were sent successfully, assume (conceptually) jobs start sync on robots.
                # Now start the AC loops for each robot involved in the synchronized job.
                checkpoints = self.current_job_recipe.get("quality_checkpoints", [])
                for job_info in jobs_info:
                    r_id = job_info["robot_id"]
                    j_name = job_info["job_name"]
                    robot_checkpoints = [cp for cp in checkpoints if cp.get("robot_id") == r_id or "robot_id" not in cp]
                    # Avoid starting duplicate AC threads if a robot is in multiple groups or steps
                    if r_id not in self._adaptive_control_threads or not self._adaptive_control_threads[r_id].is_alive():
                        ac_thread = threading.Thread(target=self._adaptive_control_loop,
                                                     args=(r_id, j_name, robot_checkpoints), daemon=True)
                        self._adaptive_control_threads[r_id] = ac_thread
                        ac_thread.start()
                        logger.info(f"Started adaptive control thread for Robot {r_id}.")
                    else:
                         logger.warning(f"Adaptive control thread for Robot {r_id} is already running.")

                all_success = True # Assume success if commands sent and AC loops started
            else:
                logger.error(f"Group action 'run_job_program_sync' failed for robots: {failed_robots}. Process halting.")
                all_success = False
                self._set_process_state(WeldingProcessState.ERROR) # Fatal error if commands failed


        else:
            logger.warning(f"Unknown group action type '{action_type}'.")
            all_success = False

        # --- Log Group Action Outcome ---
        if all_success:
            logger.info(f"Group action '{action_type}' command(s) sent successfully.")
            # Update robot statuses? Handled inside specific action logic if needed.
        else:
            logger.error(f"Group action '{action_type}' failed for one or more robots.")
            self._set_process_state(WeldingProcessState.ERROR) # Set error state if any part failed fatally
        return all_success # Return overall success status


    def _adaptive_control_loop(self, robot_id, welding_job_name, checkpoints):
        """
        Monitors welding and applies adaptive control during a specific robot's job.
        This should run in a separate thread per actively welding robot.
        Also handles real-time robot pose streaming to UE for visualization.
        """
        logger.info(f"Robot {robot_id}: Starting adaptive control and visualization stream for job '{welding_job_name}'.")
        # Set AC Active state only if the main process state is WELDING_IN_PROGRESS
        if self.current_process_state == WeldingProcessState.WELDING_IN_PROGRESS:
             self._set_process_state(WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE)

        job_start_time = time.time()
        checkpoint_idx = 0

        # --- Real-time Pose Streaming Setup (Conceptual) ---
        # Need a way to get frequent pose updates from robot_control_interface
        # This is a placeholder. Implementation depends on robot_control_interface capabilities.
        pose_stream_interval = getattr(config, 'POSE_STREAM_INTERVAL_SEC', 1 / 30.0) # Default 30 Hz


        while self._running and \
              self.current_process_state in [WeldingProcessState.WELDING_IN_PROGRESS, WeldingProcessState.ADAPTIVE_CONTROL_ACTIVE] and \
              not self.stop_event.is_set():

            # Check for job completion (this is tricky, needs robot feedback or timeout)
            # Assuming job completion signal comes from robot_control_interface or is polled.
            # For this loop's termination, we rely on self._running, stop_event, or state change.
            # A more robust loop would also check if the specific robot's job_name is still running.

            loop_start_time = time.time()

            # --- 1. Get Real-time Sensor Data ---
            # This is data aggregated by SensorDataHandler, ready for AI/Evaluation
            current_sensor_data = self.sensor_handler.get_latest_aggregated_data(robot_id)
            # logger.debug(f"Robot {robot_id}: Got sensor data for AC cycle.")

            # --- 2. Get AI Prediction (if data available and AI is ready) ---
            ai_prediction = None
            if current_sensor_data and current_sensor_data.get('values_for_ai') is not None and self.ai_engine and self.ai_engine.model_loaded:
                 try:
                    ai_prediction = self.ai_engine.process_sensor_data(current_sensor_data['values_for_ai'])
                    # Log AI prediction
                    if self.data_logger:
                         ai_pred_log_id = self.data_logger.log_ai_prediction(
                             ai_prediction,
                             input_data=current_sensor_data['values_for_ai'],
                             robot_id=robot_id,
                             job_id=self.current_job_id
                         )
                    # logger.debug(f"Robot {robot_id}: AI Prediction generated.")
                 except Exception as e:
                    logger.error(f"Robot {robot_id}: Error during AI prediction in AC loop: {e}", exc_info=True)
                    ai_prediction = None # Ensure it's None if prediction fails
            elif current_sensor_data and current_sensor_data.get('values_for_ai') is None:
                 # logger.debug(f"Robot {robot_id}: No AI input data from sensor handler this cycle.")
                 pass # Skip AI if input data is missing this cycle
            elif self.ai_engine and not self.ai_engine.model_loaded:
                 # logger.warning(f"Robot {robot_id}: AI Engine initialized but model not loaded. Skipping AI prediction.")
                 pass # Skip AI if model not loaded


            # --- 3. Evaluate Quality ---
            quality_evaluation = None
            if ai_prediction is not None or (current_sensor_data and current_sensor_data.get('values_for_evaluator')): # Evaluate if AI result or sensor data for evaluator exists
                 try:
                    quality_evaluation = self.quality_controller.evaluate_quality(
                        ai_prediction=ai_prediction,
                        real_time_sensor_data=current_sensor_data # Pass full sensor dict
                    )
                    # Log quality evaluation
                    if self.data_logger:
                         # Need the log ID of the AI prediction if it was logged
                         evaluation_log_id = self.data_logger.log_quality_evaluation(
                             quality_evaluation,
                             ai_prediction_id=ai_pred_log_id if 'ai_pred_log_id' in locals() else None, # Link to AI log
                             sensor_summary=current_sensor_data.get('values_for_evaluator'), # Log sensor summary used
                             robot_id=robot_id,
                             job_id=self.current_job_id
                         )
                    # logger.debug(f"Robot {robot_id}: Quality Evaluation generated.")
                 except Exception as e:
                    logger.error(f"Robot {robot_id}: Error during quality evaluation in AC loop: {e}", exc_info=True)
                    quality_evaluation = None # Ensure it's None if evaluation fails


            # --- 4. Generate Adaptive Adjustments ---
            adjustments = None
            if quality_evaluation and quality_evaluation.get('combined_status') not in ['Good', 'Unknown']: # Generate adjustments only if status is not Good/Unknown
                 try:
                    current_robot_params = self._get_current_robot_params(robot_id)
                    if not current_robot_params:
                         logger.warning(f"Robot {robot_id}: Cannot generate adjustments, current params not available.")
                         adjustments = None # Cannot proceed without current params
                    else:
                         adjustments = self.quality_controller.generate_adaptive_adjustments(
                             evaluation_results=quality_evaluation,
                             current_welding_params=current_robot_params # Pass current params
                         )
                         # Log generated adjustments (even if None)
                         if self.data_logger:
                              # Need the log ID of the quality evaluation if it was logged
                              self.data_logger.log_adaptive_control_action(
                                   adjustments=adjustments,
                                   triggering_eval_id=evaluation_log_id if 'evaluation_log_id' in locals() else None, # Link to evaluation log
                                   status='GENERATED' if adjustments else 'NO_ADJUSTMENT_NEEDED',
                                   robot_id=robot_id,
                                   job_id=self.current_job_id
                              )
                         # logger.debug(f"Robot {robot_id}: Adaptive adjustments generated: {adjustments}")
                 except Exception as e:
                    logger.error(f"Robot {robot_id}: Error during adjustment generation in AC loop: {e}", exc_info=True)
                    adjustments = None # Ensure it's None if generation fails


            # --- 5. Apply Adjustments ---
            if adjustments:
                logger.info(f"Robot {robot_id}: Applying adaptive adjustments: {adjustments}")
                try:
                    # Update stored current params first (these are the target values)
                    self._update_robot_params(robot_id, adjustments)

                    # Send commands to robot_control_interface to apply parameters
                    # This needs a specific method in robot_control_interface
                    # that can set multiple welding parameters (current, voltage, speed)
                    # and potentially other parameters like torch angle, speed override etc.
                    success_apply = self.robot_interface.set_welding_parameters( # Hypothetical method
                         robot_id,
                         current=adjustments.get('current', self._get_current_robot_params(robot_id).get('current')),
                         voltage=adjustments.get('voltage', self._get_current_robot_params(robot_id).get('voltage')),
                         speed=adjustments.get('speed', self._get_current_robot_params(robot_id).get('speed'))
                         # Add other parameters if applicable: torch_angle=adjustments.get('torch_angle', ...)
                    )

                    # Log application status
                    if self.data_logger:
                         # Log application status linked to the generated action
                         self.data_logger.log_adaptive_control_action(
                              adjustments=adjustments, # Log the same adjustments again
                              triggering_eval_id=evaluation_log_id if 'evaluation_log_id' in locals() else None,
                              status='APPLIED' if success_apply else 'FAILED_TO_APPLY',
                              robot_id=robot_id,
                              job_id=self.current_job_id
                         )

                    if success_apply:
                        logger.info(f"Robot {robot_id}: Adaptive adjustments applied successfully via robot interface.")
                        # Update robot status to reflect parameter change being sent
                        self._update_robot_status(robot_id, "Parameters adjusted by AC")
                    else:
                        logger.error(f"Robot {robot_id}: Failed to apply adaptive adjustments via robot interface.")
                        self._update_robot_status(robot_id, "Failed to apply AC adjustments")
                        # Decide if failure to apply adjustments is a fatal error or just a warning


                except Exception as e:
                    logger.error(f"Robot {robot_id}: Error applying adjustments: {e}", exc_info=True)
                    if self.data_logger:
                          self.data_logger.log_adaptive_control_action(
                               adjustments=adjustments,
                               triggering_eval_id=evaluation_log_id if 'evaluation_log_id' in locals() else None,
                               status='ERROR_DURING_APPLY',
                               robot_id=robot_id,
                               job_id=self.current_job_id
                          )
                    self._update_robot_status(robot_id, "Error applying AC adjustments")


            # --- 6. Real-time Robot Pose Streaming to UE (Conceptual) ---
            # Get latest robot pose from robot_control_interface or its status stream
            # Assuming robot_control_interface provides a way to get *real-time* pose
            # e.g., from a background monitoring thread/queue.
            # current_robot_pose = self.robot_interface.get_latest_pose(robot_id) # Hypothetical method
            current_robot_status = self.robot_interface.get_robot_status(robot_id) # Re-using existing polling method as placeholder

            if current_robot_status and self.physics_interface and self.physics_interface.is_connected:
                # Assuming robot status includes pose data in a format usable by send_robot_pose
                # This part needs mapping from robot status dict to send_robot_pose args
                pose_data_for_ue = None
                # Example: If status_data has 'position_tcp' and 'orientation_tcp'
                if current_robot_status.get('position_tcp') is not None:
                    pose_data_for_ue = {'tcp_transform': {'position': current_robot_status['position_tcp'], 'rotation': current_robot_status.get('orientation_tcp', [0,0,0,1])}} # Need actual rotation format
                # Example: If status_data has 'joint_angles'
                elif current_robot_status.get('joint_angles') is not None:
                    pose_data_for_ue = {'joint_angles': current_robot_status['joint_angles']}

                if pose_data_for_ue:
                    # Send the pose to UE
                    self.physics_interface.send_robot_pose(
                         robot_id=robot_id,
                         joint_angles=pose_data_for_ue.get('joint_angles'),
                         tcp_transform=pose_data_for_ue.get('tcp_transform')
                    )
                else:
                     # logger.debug(f"Robot {robot_id}: Pose data not available in robot status for UE streaming.")
                     pass # Pose data not in status or not in expected format
            elif self.physics_interface and not self.physics_interface.is_connected:
                 # logger.debug(f"Robot {robot_id}: physics_interface not connected. Skipping pose streaming.")
                 pass # Skip streaming if UE is not connected
            # else: robot_interface might not be providing status, or not initialized


            # Handle recipe-defined quality checkpoints (simplified - time-based)
            time_elapsed = time.time() - job_start_time
            if checkpoints and checkpoint_idx < len(checkpoints):
                checkpoint = checkpoints[checkpoint_idx]
                if "time_elapsed_sec" in checkpoint and time_elapsed >= checkpoint["time_elapsed_sec"]:
                    logger.info(f"Robot {robot_id}: Reached quality checkpoint {checkpoint['check_id']} at {time_elapsed:.2f}s.")
                    # Log relevant data for this checkpoint (AI prediction, evaluation, sensor data)
                    # This requires accessing the latest data collected in this AC cycle.
                    # The evaluation_log_id/ai_pred_log_id from this cycle could be used.
                    # self.data_logger.log_quality_checkpoint(...) # Need a specific log type or query previous logs by time/job/robot
                    # For now, rely on the fact that evaluation/prediction were already logged if successful this cycle.
                    checkpoint_idx += 1
                # Add logic for "job_event" based checkpoints if robot can send such events


            # Control loop frequency for AC and streaming
            # Sleep for remaining time after completing all AC and streaming tasks in this cycle
            elapsed_in_loop = time.time() - loop_start_time
            # Use a minimum interval like pose_stream_interval or a configurable AC cycle time
            loop_interval = getattr(config, 'ADAPTIVE_CONTROL_CYCLE_TIME_SEC', 1.0)
            # Ensure the loop doesn't run faster than needed for smooth streaming
            if pose_stream_interval < loop_interval:
                 loop_interval = pose_stream_interval # Run at streaming frequency if faster

            sleep_time = loop_interval - elapsed_in_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
            # else: Warning about loop taking longer than interval


        logger.info(f"Robot {robot_id}: Adaptive control and visualization stream for job '{welding_job_name}' ended.")
        # The AC thread exits when _running is False, stop_event is set, or state changes out of WELDING/ADAPTIVE.
        # Need a mechanism for this loop to know when the *robot's job* specifically finishes.
        # This could be done by monitoring robot status via robot_control_interface.

    def _process_job_thread_func(self, job_id):
        """The main logic for processing a single welding job."""
        self.current_job_id = job_id
        self.current_job_recipe = WELDING_JOBS_RECIPES.get(job_id)

        if not self.current_job_recipe:
            logger.error(f"Job ID '{job_id}' not found in recipes. Aborting.")
            self._set_process_state(WeldingProcessState.ERROR)
            # Log error event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_START_FAILED",
                     job_id=job_id,
                     details={"message": "Job ID not found in recipes"}
                 )
            return

        logger.info(f"Starting to process welding job: {job_id} - {self.current_job_recipe['description']}")
        self._set_process_state(WeldingProcessState.LOADING_PART) # Example initial state
        self.stop_event.clear() # Ensure stop event is cleared at start

        # Initialize active robots for this job
        self.active_robots = {} # {robot_id: {"current_step_index": 0, "status": "pending", "current_params": {}}}
        all_involved_robot_ids = set(self.current_job_recipe.get("robots_involved", []))
        # Add robots from group actions to involved list
        for step in self.current_job_recipe.get("steps", []):
             if "group_action" in step:
                 for r_id in step.get("robots", []):
                     all_involved_robot_ids.add(r_id)

        for robot_id in all_involved_robot_ids:
            self.active_robots[robot_id] = {"current_step_index": 0, "status": "pending", "current_params": {}}
            # Initialize current params from recipe if available, or config defaults?
            # For now, params are updated when set_welding_params action is executed.

        # --- Ensure necessary robots are connected ---
        # SystemManager is responsible for initial connection (connect_all).
        # Check if robots required for *this* job are connected.
        connected_robot_ids = self.robot_interface.connected_robot_ids if self.robot_interface else []
        required_robots_connected = True
        for robot_id in all_involved_robot_ids:
             if robot_id not in connected_robot_ids:
                 logger.error(f"Robot {robot_id} required for job {job_id} is not connected.")
                 required_robots_connected = False

        if not required_robots_connected:
             logger.error(f"Not all required robots are connected for job {job_id}. Aborting.")
             self._set_process_state(WeldingProcessState.ERROR)
             if self.data_logger:
                  self.data_logger.log_process_event(
                      event_type="JOB_START_FAILED",
                      job_id=job_id,
                      details={"message": "Not all required robots connected", "required_robots": list(all_involved_robot_ids), "connected_robots": connected_robot_ids}
                  )
             return

        # --- Ensure Physics Interface is connected if visualization/simulation is needed ---
        # SystemManager connects the physics interface. Check its status.
        # Decide if disconnected UE should abort the job or just disable visualization.
        # For now, just warn if not connected.
        if self.physics_interface and not self.physics_interface.is_connected:
             logger.warning("Physics Interface (UE) is not connected. Visualization/Simulation will be unavailable for this job.")
             if self.data_logger:
                  self.data_logger.log_process_event(
                       event_type="UE_INTERFACE_NOT_CONNECTED",
                       job_id=job_id,
                       details={"message": "UE interface not connected at job start."}
                  )


        # --- Execute Recipe Steps ---
        all_steps = self.current_job_recipe.get("steps", [])
        current_step_num = 0
        self._adaptive_control_threads = {} # Reset AC threads for this job

        while current_step_num < len(all_steps) and \
              self.current_process_state not in [WeldingProcessState.ERROR, WeldingProcessState.ABORTED] and \
              not self.stop_event.is_set():

            step_details = all_steps[current_step_num]
            logger.info(f"Job '{job_id}': Executing Step {current_step_num + 1}/{len(all_steps)}")

            success = False
            if "action" in step_details: # Single robot action
                robot_id = step_details.get("robot_id")
                # Connectivity check already done for all required robots at job start,
                # but could re-check here if connections can drop during a job.
                # if robot_id not in self.robot_interface.connected_robot_ids: ... handle runtime disconnect

                self._update_robot_status(robot_id, f"Executing step {current_step_num+1}", current_step_num)
                success = self._execute_robot_action(robot_id, step_details)

                # If this step was 'run_job_program', start adaptive control for this robot
                if success and step_details.get("action") == "run_job_program":
                    job_name = step_details.get("job_name")
                    checkpoints = self.current_job_recipe.get("quality_checkpoints", [])
                    # Filter checkpoints relevant to this robot's job if defined per robot
                    # Or if robot_id is not specified, apply to all robots in the job.
                    robot_checkpoints = [cp for cp in checkpoints if cp.get("robot_id") == robot_id or "robot_id" not in cp]

                    # Avoid starting duplicate AC threads
                    if robot_id not in self._adaptive_control_threads or not self._adaptive_control_threads[robot_id].is_alive():
                         ac_thread = threading.Thread(target=self._adaptive_control_loop,
                                                      args=(robot_id, job_name, robot_checkpoints), daemon=True)
                         self._adaptive_control_threads[robot_id] = ac_thread
                         ac_thread.start()
                         logger.info(f"Started adaptive control and visualization thread for Robot {robot_id}.")
                    else:
                         logger.warning(f"Adaptive control thread for Robot {robot_id} is already running.")


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
                         # Avoid starting duplicate AC threads
                         if r_id not in self._adaptive_control_threads or not self._adaptive_control_threads[r_id].is_alive():
                             ac_thread = threading.Thread(target=self._adaptive_control_loop,
                                                          args=(r_id, j_name, robot_checkpoints), daemon=True)
                             self._adaptive_control_threads[r_id] = ac_thread
                             ac_thread.start()
                             logger.info(f"Started adaptive control and visualization thread for Robot {r_id} in sync group.")
                         else:
                              logger.warning(f"Adaptive control thread for Robot {r_id} is already running (sync group).")

            else:
                logger.warning(f"Job '{job_id}': Step {current_step_num + 1} has no recognized action. Skipping.")
                success = True # Treat as success to proceed

            if not success:
                logger.error(f"Job '{job_id}': Failed at step {current_step_num + 1}. Process halting.")
                self._set_process_state(WeldingProcessState.ERROR)
                # Log failure event
                if self.data_logger:
                     self.data_logger.log_process_event(
                         event_type="STEP_EXECUTION_FAILED",
                         job_id=job_id,
                         details={"step_number": current_step_num + 1, "step_details": step_details}
                     )
                break # Exit step loop on failure

            current_step_num += 1
            # Add delay between steps if needed (e.g., for part loading/unloading)
            # time.sleep(getattr(config, 'INTER_STEP_DELAY_SEC', 0.1))


        # --- Wait for all adaptive control loops to finish ---
        # AC threads should ideally exit gracefully when the robot job they monitor finishes
        # or when the main process state changes to ERROR/ABORTED/COMPLETED.
        # We wait here to ensure they've cleaned up.
        if self.current_process_state not in [WeldingProcessState.ERROR, WeldingProcessState.ABORTED]:
             # If step execution finished without error, assume jobs are completed if they were started
             # A more robust system would verify job completion status from robot_interface.
             logger.info(f"Job '{job_id}': All steps executed. Setting state to COMPLETED (assuming robot jobs finished).")
             self._set_process_state(WeldingProcessState.WELDING_COMPLETED)


        logger.info(f"Waiting for adaptive control threads for job {job_id} to join...")
        threads_to_join = list(self._adaptive_control_threads.values()) # Get current AC threads
        # Clear the dict before joining to prevent issues if stop_current_job is called during join
        self._adaptive_control_threads = {}
        for thread in threads_to_join:
            if thread.is_alive():
                # Signal the AC loop within the thread to exit if it hasn't already.
                # The AC loop checks self._running, stop_event, and state.
                # If the state is now COMPLETED, ERROR, or ABORTED, the loop should exit.
                # If the loop doesn't exit, it might be stuck waiting for robot status/event.
                # Using a timeout for the join is essential.
                wait_timeout = 5 # Give it a few seconds
                logger.debug(f"Joining AC thread (timeout: {wait_timeout}s)...")
                thread.join(timeout=wait_timeout)
                if thread.is_alive():
                     logger.warning(f"Adaptive control thread {thread.name} did not terminate gracefully after join({wait_timeout}s).")
                else:
                    logger.debug(f"Adaptive control thread {thread.name} joined successfully.")


        # --- Finalize Job ---
        if self.stop_event.is_set() and self.current_process_state not in [WeldingProcessState.ERROR, WeldingProcessState.ABORTED]:
             # If stop_event was set, transition to ABORTED
             logger.info(f"Job '{job_id}' was aborted by external signal.")
             self._set_process_state(WeldingProcessState.ABORTED)
        elif self.current_process_state == WeldingProcessState.WELDING_COMPLETED:
            logger.info(f"Job '{job_id}' completed successfully.")
            # Perform any post-job actions, e.g., final quality report generation based on logged data
            # Log completion event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_COMPLETED_SUCCESS",
                     job_id=job_id,
                     details={"message": "Job completed all steps."}
                 )
        elif self.current_process_state == WeldingProcessState.ERROR:
            logger.error(f"Job '{job_id}' ended with an error state.")
            # Perform error recovery or notification
            # Log error event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_COMPLETED_ERROR",
                     job_id=job_id,
                     details={"message": "Job ended in error state."}
                 )
        elif self.current_process_state == WeldingProcessState.ABORTED:
             logger.warning(f"Job '{job_id}' ended in aborted state.")
             if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_COMPLETED_ABORTED",
                     job_id=job_id,
                     details={"message": "Job was aborted."}
                 )
        else:
            logger.warning(f"Job '{job_id}' finished with unexpected state: {self.current_process_state.name}")
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_COMPLETED_UNEXPECTED_STATE",
                     job_id=job_id,
                     details={"message": f"Job finished with unexpected state: {self.current_process_state.name}"}
                 )


        self._set_process_state(WeldingProcessState.IDLE) # Reset manager to idle
        self.current_job_id = None
        self.current_job_recipe = None
        self.active_robots = {}
        logger.info(f"Welding Process Manager returned to IDLE state.")


    def start_welding_job(self, job_id):
        """Starts a new welding job in a separate thread."""
        if self.current_process_state != WeldingProcessState.IDLE:
            logger.error(f"Cannot start job '{job_id}'. Manager is not IDLE (current state: {self.current_process_state.name}).")
            # Log event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_START_REJECTED",
                     job_id=job_id,
                     details={"message": "Manager not IDLE", "current_state": self.current_process_state.name}
                 )
            return False

        if job_id not in WELDING_JOBS_RECIPES:
             logger.error(f"Job ID '{job_id}' not found in recipes.")
             # Log event
             if self.data_logger:
                  self.data_logger.log_process_event(
                      event_type="JOB_START_REJECTED",
                      job_id=job_id,
                      details={"message": "Job ID not found in recipes"}
                  )
             return False

        logger.info(f"Queuing welding job: {job_id}")
        # Log event
        if self.data_logger:
             self.data_logger.log_process_event(
                 event_type="JOB_START_REQUESTED",
                 job_id=job_id,
                 details={"message": "Received request to start job"}
             )

        # Create and start the job processing thread
        self.process_thread = threading.Thread(target=self._process_job_thread_func, args=(job_id,), daemon=True)
        self.process_thread.start()
        logger.info(f"Job processing thread for job '{job_id}' started.")
        return True

    def stop_current_job(self):
        """Requests to stop the currently running job."""
        if self.current_process_state == WeldingProcessState.IDLE:
            logger.info("No job currently running to stop.")
             # Log event
            if self.data_logger:
                 self.data_logger.log_process_event(
                     event_type="JOB_STOP_REJECTED",
                     job_id=self.current_job_id,
                     details={"message": "No job running"}
                 )
            return False

        logger.warning(f"Attempting to stop current job: {self.current_job_id}")
         # Log event
        if self.data_logger:
             self.data_logger.log_process_event(
                 event_type="JOB_STOP_REQUESTED",
                 job_id=self.current_job_id,
                 details={"message": "Received request to stop job"}
             )

        self.stop_event.set() # Signal the process thread and AC loops to stop

        # Also send stop commands to all active robots that are likely running jobs
        # This helps the robot stop quickly.
        robots_running_jobs = [
            r_id for r_id, status in self.active_robots.items()
            if status.get("status", "").startswith("Executing step") or status.get("status", "") == "Run job command SENT" # Simplified check
        ]
        if robots_running_jobs:
             logger.info(f"Sending STOP command to active robots: {robots_running_jobs}")
             for robot_id in robots_running_jobs:
                 # Consider doing this in separate threads if stop_job can be blocking
                 # Ensure robot_interface is initialized and connected
                 if self.robot_interface and self.robot_interface._get_connection(robot_id): # Check connection
                     try:
                         self.robot_interface.stop_job(robot_id)
                         logger.info(f"STOP command sent to Robot {robot_id}.")
                         self._update_robot_status(robot_id, "STOP command sent")
                     except Exception as e:
                         logger.error(f"Error sending STOP command to Robot {robot_id}: {e}")
                         self._update_robot_status(robot_id, "STOP command failed to send")
                 else:
                     logger.warning(f"Robot {robot_id}: Robot interface not available/connected, skipping STOP command.")


        # The process thread should detect stop_event and transition to ABORTED/IDLE
        # The main loop in system_manager is responsible for waiting for this thread to join.
        logger.info(f"Stop signal sent for job {self.current_job_id}.")
        return True

    def get_manager_status(self):
        """Returns the current status of the process manager."""
        # Prepare active robots status for external view
        active_robots_summary = {}
        for r_id, status_data in self.active_robots.items():
             active_robots_summary[r_id] = {
                 "status": status_data.get("status"),
                 "current_step_index": status_data.get("current_step_index"),
                 # Do NOT include current_params here unless they are safe/needed externally
             }

        return {
            "current_job_id": self.current_job_id,
            "current_job_description": self.current_job_recipe.get("description") if self.current_job_recipe else None,
            "process_state": self.current_process_state.name,
            "active_robots": active_robots_summary,
            "process_thread_alive": self.process_thread.is_alive() if self.process_thread else False,
            "stop_event_set": self.stop_event.is_set(),
            "ac_threads_alive": {r_id: t.is_alive() for r_id, t in self._adaptive_control_threads.items()}
        }

# Example Usage (Requires SystemManager to instantiate and provide dependencies)
# This module is not typically run directly, but orchestrated by SystemManager.
# The __main__ block in system_manager.py provides the example run for the whole system.