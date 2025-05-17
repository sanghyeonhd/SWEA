# src/quality_evaluator_adaptive_control.py
# Description: (MODIFIED) Evaluates weld quality based on AI predictions (and other sources),
#              and generates real-time adaptive control adjustments.
#              Consumes AI predictions from MQ, publishes control commands/evaluations to MQ.

import json
import time
import threading
import logging
import queue

# Import Message Queue client (e.g., pika for RabbitMQ)
try:
    import pika
    pika_available = True
except ImportError:
    pika = None

from src import config # Import the main config module
# from src.data_logger_db import DataLoggerDB # For logging evaluations and control actions
# from src.sensor_data_handler import SensorDataHandler # To get latest sensor data if not all in AI pred message
# from src.physics_interface import UnrealSimulatorInterface # To get physics sim results if used in ensemble

# Setup logging (use config's logger if available, else basic)
logger = config.logging.getLogger(__name__) if hasattr(config, 'logging') else logging.getLogger(__name__)
if not hasattr(config, 'logging'):
    logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.INFO)


# Re-define quality levels (could be loaded from config or shared module)
QUALITY_CLASSES = getattr(config, 'QUALITY_CLASSES_MAP', {
    0: "Complete Fusion / Good",
    1: "Incomplete Fusion / Lack of Fusion",
    2: "Undercut",
    3: "Hot Tear / Crack"
})

# Adaptive Control Rules (loaded from config or external file/DB)
# Example: config.ADAPTIVE_CONTROL_RULES (loaded by config.py from a JSON)
ADAPTIVE_CONTROL_RULES = config.ADAPTIVE_CONTROL_RULES if hasattr(config, 'ADAPTIVE_CONTROL_RULES') else {}
if not ADAPTIVE_CONTROL_RULES:
    logger.warning("ADAPTIVE_CONTROL_RULES are not defined or empty in config. Adaptive control will be limited.")


class QualityEvaluatorAdaptiveControl:
    """
    Evaluates weld quality from AI predictions and other data sources,
    and determines adaptive control actions, publishing results/commands to MQ.
    """
    _instance = None # Singleton pattern
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_logger=None, sensor_handler=None, physics_interface=None):
        with self._lock:
            if self._initialized: return
            self._initialized = True

            logger.info("Quality Evaluator & Adaptive Control (Advanced) initializing...")
            self.data_logger = data_logger
            self.sensor_handler = sensor_handler # To fetch current sensor state if needed
            self.physics_interface = physics_interface # To fetch physics sim results if needed

            # --- Message Queue Setup ---
            self._mq_connection = None
            self._mq_channel = None # Use one channel for both consuming and publishing for simplicity here
            self._mq_lock = threading.Lock() # For thread-safe MQ operations
            self._mq_consumer_thread = None
            self._stop_mq_event = threading.Event()
            self._incoming_prediction_queue = queue.Queue(maxsize=getattr(config, 'QEA_INPUT_QUEUE_SIZE', 100))
            self._evaluation_worker_thread = None

            if config.USE_MESSAGE_QUEUE and config.MQ_TYPE == 'rabbitmq':
                if not pika:
                    logger.error("Pika library not found for RabbitMQ. MQ features for QEA disabled.")
                else:
                    if self._connect_mq():
                        logger.info("QEA: MQ Connection established.")
                    else:
                        logger.error("QEA: Failed to connect to MQ at initialization.")
            elif config.USE_MESSAGE_QUEUE:
                logger.warning(f"MQ type '{config.MQ_TYPE}' not supported for QEA yet.")


    def _connect_mq(self):
        """Establishes connection to RabbitMQ and declares exchanges/queues."""
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika: return False
        with self._mq_lock:
            if self._mq_channel and self._mq_channel.is_open: return True
            try:
                logger.info(f"QEA: Connecting MQ to {config.MQ_HOST}:{config.MQ_PORT}")
                credentials = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
                parameters = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, credentials, heartbeat=600)
                self._mq_connection = pika.BlockingConnection(parameters)
                self._mq_channel = self._mq_connection.channel()

                # Declare exchange to consume AI predictions from
                self._mq_channel.exchange_declare(exchange=config.MQ_AI_PREDICTION_EXCHANGE, exchange_type='topic', durable=True)
                # Declare an exclusive queue for this QEA instance
                result = self._mq_channel.queue_declare(queue='', exclusive=True)
                self.ai_prediction_queue_name = result.method.queue
                # Bind to AI prediction topics (e.g., "ai.prediction.#")
                # Routing key might be more specific if needed e.g. "ai.prediction.robot*.quality_class"
                ai_pred_binding_key = "ai.prediction.#"
                self._mq_channel.queue_bind(exchange=config.MQ_AI_PREDICTION_EXCHANGE, queue=self.ai_prediction_queue_name, routing_key=ai_pred_binding_key)
                logger.info(f"QEA: MQ Consumer queue '{self.ai_prediction_queue_name}' bound to '{ai_pred_binding_key}'.")

                # Declare exchange to publish evaluation results and control commands to
                # Could be separate exchanges or one with different routing keys
                self.results_exchange_name = getattr(config, 'MQ_QEA_RESULTS_EXCHANGE', 'qea_results_exchange')
                self._mq_channel.exchange_declare(exchange=self.results_exchange_name, exchange_type='topic', durable=True)
                logger.info(f"QEA: MQ Publisher exchange '{self.results_exchange_name}' declared.")
                return True
            except Exception as e:
                logger.error(f"QEA: Failed to connect MQ: {e}", exc_info=True)
                if self._mq_connection and self._mq_connection.is_open: self._mq_connection.close()
                self._mq_connection, self._mq_channel = None, None
                return False

    def _disconnect_mq(self):
        """Closes RabbitMQ connection."""
        with self._mq_lock:
            if self._mq_channel and self._mq_channel.is_open:
                try: self._mq_channel.close()
                except Exception as e_ch: logger.error(f"QEA: Error closing MQ channel: {e_ch}")
            if self._mq_connection and self._mq_connection.is_open:
                try: self._mq_connection.close()
                except Exception as e_conn: logger.error(f"QEA: Error closing MQ connection: {e_conn}")
            self._mq_channel, self._mq_connection = None, None
            logger.info("QEA: MQ connection closed.")


    def _publish_qea_output(self, routing_key_suffix, output_data_dict):
        """Publishes evaluation results or control commands to the QEA results exchange."""
        if not config.USE_MESSAGE_QUEUE or not (self._mq_channel and self._mq_channel.is_open):
            logger.warning(f"QEA: MQ channel not open for publishing output. Data for {routing_key_suffix} not sent.")
            # Optionally try to reconnect: if not self._connect_mq(): return False
            return False
        try:
            # Example routing_key: "qea.evaluation.robot1" or "qea.control_command.robot1"
            full_routing_key = f"qea.{routing_key_suffix}"
            message_body_str = json.dumps(output_data_dict, ensure_ascii=False)
            with self._mq_lock:
                self._mq_channel.basic_publish(
                    exchange=self.results_exchange_name,
                    routing_key=full_routing_key,
                    body=message_body_str,
                    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE, content_type='application/json')
                )
            # logger.debug(f"QEA Published to MQ: Key='{full_routing_key}', Body='{message_body_str[:100]}...'")
            return True
        except Exception as e:
            logger.error(f"QEA: Failed to publish output to MQ. RoutingKeySuffix: {routing_key_suffix}. Error: {e}", exc_info=True)
            if isinstance(e, (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker)):
                 self._disconnect_mq() # Disconnect on connection error
            return False


    def _mq_ai_prediction_callback(self, ch, method, properties, body):
        """Callback for processing AI prediction messages received from MQ."""
        try:
            # logger.debug(f"QEA MQ Callback: Received message from {method.routing_key}")
            ai_prediction_packet = json.loads(body.decode('utf-8'))
            # Packet from AIInferenceEngine:
            # { "timestamp_utc_prediction": ..., "timestamp_utc_sensor": ...,
            #   "robot_id": ..., "sensor_id": ..., "prediction": {...}, "source_system": "AIInferenceEngine" }

            # Put the AI prediction packet onto the internal processing queue
            self._incoming_prediction_queue.put(ai_prediction_packet, timeout=0.5)

        except json.JSONDecodeError:
            logger.error(f"QEA MQ Callback: Failed to decode JSON from AI prediction message: {body.decode('utf-8')[:200]}")
        except queue.Full:
            logger.warning("QEA input prediction queue is full. AI prediction from MQ might be dropped.")
        except Exception as e:
            logger.error(f"QEA MQ Callback: Error processing AI prediction message: {e}", exc_info=True)


    def _evaluation_worker_loop(self):
        """Worker thread to process AI predictions from internal queue, evaluate, and generate control."""
        logger.info("QEA evaluation worker thread started. Waiting for AI predictions...")
        while not self._stop_mq_event.is_set():
            try:
                ai_prediction_packet = self._incoming_prediction_queue.get(timeout=1.0) # Wait 1 sec

                robot_id = ai_prediction_packet.get('robot_id')
                ai_prediction_result = ai_prediction_packet.get('prediction')
                sensor_timestamp_utc = ai_prediction_packet.get('timestamp_utc_sensor') # Timestamp of sensor data used for this AI pred

                if not ai_prediction_result:
                    logger.warning(f"QEA Worker: Received packet with no 'prediction' field from robot {robot_id}. Skipping.")
                    self._incoming_prediction_queue.task_done()
                    continue

                # --- 1. Gather Additional Data (Sensor, Physics Sim) ---
                # SensorDataHandler might provide more context or non-AI features.
                # This requires careful synchronization if AI prediction and latest sensor state are from different times.
                # Assume for now, AI prediction packet contains features or we fetch based on its sensor_timestamp_utc
                real_time_sensor_data_for_eval = None
                if self.sensor_handler:
                    # Option A: Get sensor data matching the timestamp of AI prediction input (complex to implement)
                    # Option B: Get the *absolute* latest sensor data (might be slightly out of sync with AI pred)
                    # Option C: Assume AI prediction packet has all sensor info needed for rule-based part of evaluation
                    # For now, assume AI prediction packet is self-contained for this simple example,
                    # or that we fetch latest as a supplement.
                    # real_time_sensor_data_for_eval = self.sensor_handler.get_latest_aggregated_data(robot_id)
                    pass # Placeholder for fetching more sensor data if needed

                physics_sim_result = None
                # if self.physics_interface and config.USE_PHYSICS_IN_EVALUATION:
                #     # Request a physics simulation result relevant to the current state/prediction
                #     # This would likely be an asynchronous request to UE.
                #     physics_sim_result = self.physics_interface.run_simulation(...) # Needs relevant params


                # --- 2. Evaluate Quality ---
                evaluation_output = self.evaluate_quality(
                    ai_prediction=ai_prediction_result,
                    real_time_sensor_data=real_time_sensor_data_for_eval, # From sensor_handler
                    physics_prediction=physics_sim_result # From physics_interface
                )
                evaluation_output['robot_id'] = robot_id # Add robot_id for routing/logging
                evaluation_output['triggering_ai_prediction_ts'] = ai_prediction_packet.get('timestamp_utc_prediction')
                evaluation_output['triggering_sensor_ts'] = sensor_timestamp_utc

                # Log evaluation result (can be done by a separate logger service subscribing to QEA outputs)
                if self.data_logger:
                     eval_log_id = self.data_logger.log_quality_evaluation(evaluation_output, robot_id=robot_id, job_id=ai_prediction_packet.get('job_id'))


                # --- 3. Publish Evaluation Result to MQ ---
                eval_routing_key_suffix = f"evaluation.{'robot'+str(robot_id) if robot_id else 'global'}.{evaluation_output.get('combined_status','unknown').lower()}"
                self._publish_qea_output(eval_routing_key_suffix, evaluation_output)


                # --- 4. Generate and Publish Adaptive Control Adjustments ---
                if evaluation_output.get('combined_status') not in ['Good', 'Unknown']: # Or other conditions
                    current_welding_params = None
                    # How to get current_welding_params?
                    # Option 1: If WeldingProcessManager publishes them to MQ or status store.
                    # Option 2: If AI prediction packet includes the input params used for that prediction.
                    # Option 3: If this module tracks them (less ideal, duplication).
                    # For this example, assume it might be part of ai_prediction_packet or fetched.
                    # This is a critical piece of information for generating adjustments.
                    # Let's assume it might be in `ai_prediction_packet['context']['current_params']` (placeholder)
                    current_welding_params = ai_prediction_packet.get('context', {}).get('current_params')
                    if not current_welding_params and self.robot_interface: # Fallback: try to get from robot_interface (might be stale)
                         status = self.robot_interface.get_latest_robot_status(robot_id)
                         if status: current_welding_params = status.get('welding_parameters') # Assuming this key exists

                    if current_welding_params:
                        adjustments = self.generate_adaptive_adjustments(
                            evaluation_results=evaluation_output,
                            current_welding_params=current_welding_params
                        )
                        if adjustments:
                            logger.info(f"Robot {robot_id}: Generated adaptive adjustments: {adjustments}")
                            # Publish control command to MQ for WeldingProcessManager to pick up
                            control_cmd_packet = {
                                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "robot_id": robot_id,
                                "job_id": ai_prediction_packet.get('job_id'), # Pass job_id if available
                                "command_type": "adaptive_parameter_adjustment",
                                "adjustments": adjustments,
                                "triggering_evaluation": evaluation_output # Include full eval for context
                            }
                            adj_routing_key_suffix = f"control_command.robot{robot_id}.parameter_adjustment"
                            self._publish_qea_output(adj_routing_key_suffix, control_cmd_packet)

                            # Log control action (can also be done by WPM when it applies it)
                            if self.data_logger:
                                 self.data_logger.log_adaptive_control_action(
                                     adjustments,
                                     triggering_eval_id=eval_log_id if 'eval_log_id' in locals() else None,
                                     status='GENERATED_FOR_MQ', robot_id=robot_id, job_id=ai_prediction_packet.get('job_id')
                                 )
                        else:
                             logger.debug(f"Robot {robot_id}: No adaptive adjustments generated for current evaluation.")
                    else:
                        logger.warning(f"Robot {robot_id}: Cannot generate adjustments. Current welding parameters unknown.")

                self._incoming_prediction_queue.task_done()

            except queue.Empty:
                pass # Timeout, check stop_event and continue
            except Exception as e:
                logger.error(f"QEA evaluation worker thread error: {e}", exc_info=True)
                time.sleep(0.5)

        logger.info("QEA evaluation worker thread stopped.")


    def start_consuming_predictions(self):
        """Starts the MQ subscriber thread to consume AI predictions and the evaluation worker."""
        if not config.USE_MESSAGE_QUEUE or config.MQ_TYPE != 'rabbitmq' or not pika:
            logger.warning("MQ not enabled or Pika not available. QEA will not consume AI predictions from MQ.")
            return

        if self._mq_consumer_thread and self._mq_consumer_thread.is_alive():
            logger.info("QEA MQ AI prediction consumer already running.")
            return

        if not (self._mq_channel and self._mq_channel.is_open):
            if not self._connect_mq():
                logger.error("Cannot start AI prediction consumer: Failed to connect to MQ.")
                return

        logger.info("Starting QEA MQ AI prediction consumer thread...")
        self._stop_mq_event.clear()
        self._mq_consumer_thread = threading.Thread(target=self._mq_consumer_main_loop, name="QEAMQConsumer", daemon=True)
        self._mq_consumer_thread.start()

        if self._evaluation_worker_thread is None or not self._evaluation_worker_thread.is_alive():
             self._evaluation_worker_thread = threading.Thread(target=self._evaluation_worker_loop, name="QAEvalWorker", daemon=True)
             self._evaluation_worker_thread.start()


    def _mq_consumer_main_loop(self):
        """Main loop for the MQ consumer thread."""
        logger.info("QEA MQ consumer main loop started.")
        while not self._stop_mq_event.is_set():
            try:
                if not (self._mq_channel and self._mq_channel.is_open):
                    logger.warning("QEA: MQ channel not open in consumer loop. Attempting reconnect...")
                    if not self._connect_mq():
                        logger.error("QEA: MQ consumer re-connection failed. Waiting.")
                        time.sleep(5); continue

                logger.info(f"QEA: Starting to consume AI predictions from MQ queue '{self.ai_prediction_queue_name}'...")
                self._mq_channel.basic_consume(queue=self.ai_prediction_queue_name, on_message_callback=self._mq_ai_prediction_callback, auto_ack=True)
                self._mq_channel.start_consuming() # Blocking call
                logger.info("QEA: MQ consumption stopped.")
                if self._stop_mq_event.is_set(): break

            except pika.exceptions.ConnectionClosedByBroker: logger.warning("QEA: MQ Connection closed by broker. Reconnecting..."); self._disconnect_mq(); time.sleep(5)
            except pika.exceptions.AMQPChannelError as e: logger.error(f"QEA: MQ Channel error: {e}. Re-establishing..."); self._disconnect_mq(); time.sleep(5)
            except pika.exceptions.AMQPConnectionError as e: logger.error(f"QEA: MQ Connection error: {e}. Reconnecting..."); self._disconnect_mq(); time.sleep(5)
            except Exception as e: logger.error(f"QEA: MQ consumer main loop unexpected error: {e}", exc_info=True); self._disconnect_mq(); time.sleep(5)
        logger.info("QEA MQ consumer main loop stopped.")


    def stop_consuming_predictions(self):
        """Stops the MQ subscriber thread and evaluation worker."""
        logger.info("QEA: Stopping AI prediction consumption and evaluation worker...")
        self._stop_mq_event.set()

        if self._mq_consumer_thread and self._mq_consumer_thread.is_alive():
            if self._mq_channel and self._mq_channel.is_open:
                try: self._mq_channel.stop_consuming() # Request consumer to stop
                except Exception as e: logger.error(f"QEA: Error stopping MQ consumer channel: {e}")
            logger.info("Waiting for QEA MQ consumer thread to join...")
            self._mq_consumer_thread.join(timeout=5.0)
            if self._mq_consumer_thread.is_alive(): logger.warning("QEA MQ consumer thread did not join.")
        self._mq_consumer_thread = None

        if self._evaluation_worker_thread and self._evaluation_worker_thread.is_alive():
            logger.info("Waiting for QEA evaluation worker thread to join...")
            self._evaluation_worker_thread.join(timeout=5.0)
            if self._evaluation_worker_thread.is_alive(): logger.warning("QEA evaluation worker thread did not join.")
        self._evaluation_worker_thread = None

        self._disconnect_mq()
        logger.info("QEA: AI prediction consumption and evaluation worker stopped.")


    # --- Core Evaluation and Adjustment Logic (similar to previous version, with enhancements) ---
    def evaluate_quality(self, ai_prediction, real_time_sensor_data=None, physics_prediction=None):
        """
        Evaluates weld quality based on AI, sensors, and optionally physics simulation.
        (This is a simplified version. Real ensemble and XAI would be more complex).
        """
        evaluation_results = {
            'timestamp_evaluation_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'ai_evaluation': None,
            'sensor_rule_evaluation': {},
            'physics_evaluation': None,
            'combined_status': 'Unknown', # Good, Warning, Critical, Acceptable
            'detected_issues': [],
            'uncertainty_score': None, # Placeholder for AI uncertainty
            'explanation_summary': None # Placeholder for XAI output
        }

        # 1. AI Prediction Evaluation
        if ai_prediction:
            ai_eval = {}
            if 'predicted_class' in ai_prediction: # Classification
                pred_class = ai_prediction['predicted_class']
                ai_eval['quality_level_predicted'] = QUALITY_CLASSES.get(pred_class, f"Unknown Class ({pred_class})")
                ai_eval['confidence'] = ai_prediction.get('confidence')
                # Example: if confidence is low, it might increase uncertainty or lower overall score
                if ai_eval['confidence'] is not None and ai_eval['confidence'] < getattr(config, 'QEA_MIN_AI_CONFIDENCE_THRESHOLD', 0.5):
                     evaluation_results['detected_issues'].append(f"Low AI Confidence ({ai_eval['confidence']:.2f}) for {ai_eval['quality_level_predicted']}")
                if pred_class != 0 : # Assuming class 0 is 'Good'
                    evaluation_results['detected_issues'].append(f"AI_Defect: {ai_eval['quality_level_predicted']}")
            elif 'predicted_score' in ai_prediction: # Regression
                score = ai_prediction['predicted_score']
                ai_eval['numeric_score_predicted'] = score
                if score > 0.9: ai_eval['quality_level_predicted'] = "Excellent"
                elif score > 0.75: ai_eval['quality_level_predicted'] = "Good"
                elif score > 0.5: ai_eval['quality_level_predicted'] = "Acceptable"
                else: ai_eval['quality_level_predicted'] = "Poor / Defect Likely"; evaluation_results['detected_issues'].append("AI_PoorScore")
            # Store raw prediction
            ai_eval['raw_prediction'] = ai_prediction
            evaluation_results['ai_evaluation'] = ai_eval
            # Conceptual: If AI provides uncertainty, store it
            # evaluation_results['uncertainty_score'] = ai_prediction.get('uncertainty')
            # Conceptual: If XAI provides explanation
            # evaluation_results['explanation_summary'] = ai_prediction.get('explanation')


        # 2. Sensor Rule-Based Evaluation (Using real_time_sensor_data from SensorDataHandler via WPM or direct subscription)
        # This is where real-time sensor data is directly checked against rules.
        # The 'real_time_sensor_data' should be a dict of sensor_name: value, timestamped.
        if real_time_sensor_data:
            # Example: if real_time_sensor_data is {'temperature': 450, 'arc_stability': 0.4}
            temp = real_time_sensor_data.get('temperature', {}).get('value') # Assuming sensor data packet format
            arc_stab = real_time_sensor_data.get('arc_stability_index', {}).get('value')
            if temp is not None and temp > getattr(config, 'SENSOR_TEMP_THRESHOLD_CRITICAL', 450):
                evaluation_results['sensor_rule_evaluation']['high_temperature'] = temp
                evaluation_results['detected_issues'].append("SENSOR_HighTemp")
            if arc_stab is not None and arc_stab < getattr(config, 'SENSOR_ARC_STABILITY_THRESHOLD_WARN', 0.6):
                evaluation_results['sensor_rule_evaluation']['low_arc_stability'] = arc_stab
                evaluation_results['detected_issues'].append("SENSOR_LowArcStability")


        # 3. Physics Simulation Evaluation (If physics_prediction is provided and enabled)
        if physics_prediction and getattr(config, 'USE_PHYSICS_IN_EVALUATION', False):
            evaluation_results['physics_evaluation'] = physics_prediction # Store raw physics prediction
            # Example: if physics_prediction is {'quality_score': 0.55, 'predicted_bead_shape': 'Concave'}
            phys_score = physics_prediction.get('quality_score')
            if phys_score is not None and phys_score < getattr(config, 'PHYSICS_SCORE_THRESHOLD_WARN', 0.6):
                evaluation_results['detected_issues'].append(f"PHYSICS_PoorScore ({phys_score:.2f})")


        # 4. Combine Evaluations & Determine Overall Status (Simplified Ensemble/Priority Logic)
        # This logic needs to be domain-specific and robust.
        # Example: Critical sensor rules override AI. AI defect overrides good physics.
        if "SENSOR_HighTemp" in evaluation_results['detected_issues'] or "SENSOR_LowArcStability" in evaluation_results['detected_issues']: # Critical sensor issues
            evaluation_results['combined_status'] = 'Critical'
        elif any(issue.startswith("AI_Defect") or issue.startswith("AI_PoorScore") for issue in evaluation_results['detected_issues']):
            evaluation_results['combined_status'] = 'Warning' # Or Critical based on AI confidence/severity
        elif any(issue.startswith("PHYSICS_PoorScore") for issue in evaluation_results['detected_issues']):
             evaluation_results['combined_status'] = 'Warning' # If physics predicts poor
        elif not evaluation_results['detected_issues']: # No issues detected
            evaluation_results['combined_status'] = 'Good'
        else: # Other minor issues or low confidence might be 'Acceptable' or 'Warning'
             evaluation_results['combined_status'] = 'Acceptable' # Default if some non-critical issues


        logger.info(f"Quality Evaluation (Robot {evaluation_results.get('robot_id', 'N/A')}): Status={evaluation_results['combined_status']}, Issues={evaluation_results['detected_issues']}")
        return evaluation_results


    def generate_adaptive_adjustments(self, evaluation_results, current_welding_params):
        """Generates adaptive control adjustments based on evaluation and rules."""
        if not ADAPTIVE_CONTROL_RULES: return None # No rules defined
        if not current_welding_params:
            logger.warning("Cannot generate adjustments: current_welding_params not provided.")
            return None

        suggested_adjustments = {} # Stores {param_name: target_absolute_value}
        status = evaluation_results.get('combined_status')
        # Prioritize issues detected by direct sensor rules or high-confidence AI
        # This example uses a simple loop through all detected_issues.
        # A real system might have a priority for issues.

        # Create a mutable copy of current params to calculate target values
        target_params = current_welding_params.copy()

        if status not in ['Good', 'Excellent', 'Acceptable', 'Unknown']: # Adjust only if not good/unknown
            logger.info(f"Status is '{status}'. Evaluating adaptive control rules...")
            applied_rules_for_params = set() # Track which params have already been adjusted by a rule

            for issue_code in evaluation_results.get('detected_issues', []):
                # Map issue_code (e.g., "AI_Defect: Undercut", "SENSOR_HighTemp") to a rule key
                # This mapping needs to be robust. For now, try direct match and then parts of it.
                rule_key_to_find = issue_code
                if issue_code.startswith("AI_Defect: "): rule_key_to_find = issue_code.replace("AI_Defect: ", "")
                elif issue_code.startswith("SENSOR_"): rule_key_to_find = issue_code # Example: "SENSOR_HighTemp" rule key

                if rule_key_to_find in ADAPTIVE_CONTROL_RULES:
                    rules = ADAPTIVE_CONTROL_RULES[rule_key_to_find]
                    logger.debug(f" Applying rules for issue/key: '{rule_key_to_find}' -> {rules}")
                    for param_to_adjust, adjustment_value_str in rules.items():
                        if param_to_adjust in applied_rules_for_params:
                             logger.debug(f" Parameter '{param_to_adjust}' already adjusted by a previous rule this cycle. Skipping.")
                             continue # Avoid multiple rules adjusting the same param in one cycle

                        if param_to_adjust not in current_welding_params:
                            logger.warning(f"  Rule wants to adjust '{param_to_adjust}', but it's not in current_welding_params.")
                            continue

                        current_val = float(target_params[param_to_adjust]) # Use the evolving target_params for calculation
                        target_val = current_val # Initialize with current value

                        # --- Parse adjustment_value_str ---
                        # Example: "+10" (absolute offset), "-5%" (percentage), "150" (absolute set)
                        try:
                            if isinstance(adjustment_value_str, (int, float)): # Direct value
                                target_val = float(adjustment_value_str)
                            elif adjustment_value_str.endswith('%'):
                                percentage = float(adjustment_value_str[:-1])
                                target_val = current_val * (1 + percentage / 100.0)
                            elif adjustment_value_str.startswith('+') or adjustment_value_str.startswith('-'):
                                offset = float(adjustment_value_str)
                                target_val = current_val + offset
                            else: # Assume it's an absolute value to set
                                target_val = float(adjustment_value_str)
                        except ValueError:
                            logger.error(f"  Invalid adjustment format '{adjustment_value_str}' for param '{param_to_adjust}'. Skipping.")
                            continue

                        # --- Apply Min/Max Limits from config.PARAM_RANGES / config.ADJUSTMENT_PARAMS ---
                        # Combine both dicts for limits lookup
                        all_param_limits = {**getattr(config, 'PARAM_RANGES', {}), **getattr(config, 'ADJUSTMENT_PARAMS', {})}
                        min_limit, max_limit = all_param_limits.get(param_to_adjust, (None, None))

                        if min_limit is not None and target_val < min_limit: target_val = min_limit
                        if max_limit is not None and target_val > max_limit: target_val = max_limit

                        # --- Apply Max Relative Change per Step (Example) ---
                        # This prevents overly aggressive single-step changes.
                        # max_relative_change = getattr(config, 'AC_MAX_RELATIVE_CHANGE_PERCENT', 5.0) / 100.0
                        # if abs(target_val - current_val) > abs(current_val * max_relative_change) and current_val != 0:
                        #     sign = 1 if target_val > current_val else -1
                        #     target_val = current_val + sign * abs(current_val * max_relative_change)
                        #     # Re-apply hard limits
                        #     if min_limit is not None: target_val = max(target_val, min_limit)
                        #     if max_limit is not None: target_val = min(target_val, max_limit)


                        # Store the calculated target absolute value
                        # Only add to suggested_adjustments if it's different from current_params
                        # to avoid sending redundant commands.
                        # However, WPM might want the full set of "target" params.
                        # For this example, we send only changed params as "adjustments"
                        # but the actual implementation in WPM should decide how to apply.
                        # Let's assume we update the target_params dict and then see what changed.
                        if abs(target_val - float(current_welding_params[param_to_adjust])) > 1e-3 : # Check for significant change
                             target_params[param_to_adjust] = round(target_val, 2) # Update the evolving target
                             applied_rules_for_params.add(param_to_adjust)
                             logger.info(f"  Adjustment for '{param_to_adjust}': From {current_welding_params[param_to_adjust]} to {target_params[param_to_adjust]:.2f}")

            # Collect all parameters that actually changed from original current_welding_params
            for param_name, final_target_value in target_params.items():
                original_value = float(current_welding_params.get(param_name, final_target_value)) # Handle new params added by rule
                if abs(final_target_value - original_value) > 1e-3: # If significantly changed
                     suggested_adjustments[param_name] = final_target_value


        if not suggested_adjustments:
            logger.info("No adaptive adjustments generated or needed based on current rules and evaluation.")
            return None

        logger.info(f"Final suggested_adjustments to send: {suggested_adjustments}")
        return suggested_adjustments


    def start(self):
        """Starts the MQ consumer thread and evaluation worker thread."""
        self.start_consuming_predictions()

    def stop(self):
        """Stops the MQ consumer thread and evaluation worker thread."""
        self.stop_consuming_predictions()


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    logger.info("--- Quality Evaluator & Adaptive Control (Advanced with MQ) Example ---")

    # Dummy Config
    class DummyConfigQEA:
        USE_MESSAGE_QUEUE = True
        MQ_TYPE = 'rabbitmq'
        MQ_HOST = 'localhost'; MQ_PORT = 5672; MQ_USER = 'guest'; MQ_PASSWORD = 'guest'; MQ_VHOST = '/'
        MQ_AI_PREDICTION_EXCHANGE = 'ai_predictions_exchange_test_qea'
        MQ_QEA_RESULTS_EXCHANGE = 'qea_results_exchange_test_qea' # For publishing results
        # ADAPTIVE_CONTROL_RULES needs to be populated for generate_adaptive_adjustments to do something
        ADAPTIVE_CONTROL_RULES = {
            "AI_Defect: Undercut": {"voltage": "-0.5", "speed": "+5%"}, # Example rule
            "SENSOR_HighTemp": {"current": "-10"}
        }
        PARAM_RANGES = {'current': (80,200), 'voltage': (15,30), 'speed': (100,500)} # For limits
        ADJUSTMENT_PARAMS = {}
        # For dummy SensorDataHandler
        SENSOR_CONFIGS = [] # Not used by QEA directly if sensor_handler is a mock
        # For dummy DataLogger
        # DATABASE_PATH = 'test_qea_log.db'
        LOG_LEVEL = 'DEBUG'


    config = DummyConfigQEA() # Override global config for this test

    # Mock dependencies
    class MockDataLogger:
        def log_quality_evaluation(self, *args, **kwargs): logger.debug(f"MockLog QualityEval: {args} {kwargs}")
        def log_adaptive_control_action(self, *args, **kwargs): logger.debug(f"MockLog AdaptiveControl: {args} {kwargs}")
    mock_logger = MockDataLogger()

    # --- Initialize QEA ---
    qea_instance = QualityEvaluatorAdaptiveControl(data_logger=mock_logger)

    if config.USE_MESSAGE_QUEUE and pika:
        qea_instance.start() # Starts MQ consumer and worker thread

        # --- Simulate AIInferenceEngine publishing AI predictions to MQ ---
        mq_pub_conn = None
        mq_pub_channel = None
        try:
            logger.info("[TestAIPublisher] Connecting to RabbitMQ to send dummy AI predictions...")
            creds = pika.PlainCredentials(config.MQ_USER, config.MQ_PASSWORD)
            params = pika.ConnectionParameters(config.MQ_HOST, config.MQ_PORT, config.MQ_VHOST, creds)
            mq_pub_conn = pika.BlockingConnection(params)
            mq_pub_channel = mq_pub_conn.channel()
            mq_pub_channel.exchange_declare(exchange=config.MQ_AI_PREDICTION_EXCHANGE, exchange_type='topic', durable=True)
            logger.info("[TestAIPublisher] Connected.")

            # Example AI Prediction Packet 1 (Good)
            ai_pred_packet_1 = {
                "timestamp_utc_prediction": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "timestamp_utc_sensor": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(milliseconds=100)).isoformat(),
                "robot_id": 1, "sensor_id": "aggregated_robot1",
                "prediction": {'predicted_class': 0, 'confidence': 0.95}, # Good weld
                "source_system": "AIInferenceEngine",
                "context": {"current_params": {'current': 150, 'voltage': 22, 'speed': 300}} # Current welding params
            }
            routing_key_1 = "ai.prediction.robot1.class_0"
            mq_pub_channel.basic_publish(exchange=config.MQ_AI_PREDICTION_EXCHANGE, routing_key=routing_key_1, body=json.dumps(ai_pred_packet_1))
            logger.info(f"[TestAIPublisher] Sent GOOD AI prediction for Robot 1.")

            time.sleep(1) # Give QEA time to process

            # Example AI Prediction Packet 2 (Defect)
            ai_pred_packet_2 = {
                "timestamp_utc_prediction": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "timestamp_utc_sensor": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(milliseconds=90)).isoformat(),
                "robot_id": 1, "sensor_id": "aggregated_robot1",
                "prediction": {'predicted_class': 2, 'confidence': 0.85}, # Undercut
                "source_system": "AIInferenceEngine",
                "context": {"current_params": {'current': 160, 'voltage': 25, 'speed': 280}}
            }
            routing_key_2 = "ai.prediction.robot1.class_2"
            mq_pub_channel.basic_publish(exchange=config.MQ_AI_PREDICTION_EXCHANGE, routing_key=routing_key_2, body=json.dumps(ai_pred_packet_2))
            logger.info(f"[TestAIPublisher] Sent DEFECT (Undercut) AI prediction for Robot 1.")

            logger.info("Waiting for QEA to process (approx 5s)...Published results/commands can be seen in RabbitMQ Management UI.")
            time.sleep(5) # Let QEA worker and publisher do their thing

        except Exception as e_test_pub:
            logger.error(f"[TestAIPublisher] Error: {e_test_pub}")
        finally:
            if mq_pub_channel and mq_pub_channel.is_open: mq_pub_channel.close()
            if mq_pub_conn and mq_pub_conn.is_open: mq_pub_conn.close()
            logger.info("[TestAIPublisher] Connection closed.")

    else:
        logger.warning("MQ disabled or Pika not available. Skipping MQ test for QEA.")
        # Direct call test (if MQ is off)
        logger.info("\n--- Testing QEA Direct Call (No MQ) ---")
        ai_pred_direct = {'predicted_class': 2, 'confidence': 0.8} # Undercut
        current_params_direct = {'current': 160, 'voltage': 25, 'speed': 280}
        eval_direct = qea_instance.evaluate_quality(ai_prediction=ai_pred_direct)
        logger.info(f"Direct Evaluation Output: {eval_direct}")
        if eval_direct:
             adjustments_direct = qea_instance.generate_adaptive_adjustments(eval_direct, current_params_direct)
             logger.info(f"Direct Adjustments Generated: {adjustments_direct}")


    # --- Shutdown ---
    logger.info("\nShutting down Quality Evaluator & Adaptive Control...")
    qea_instance.stop() # Stops MQ consumer and worker

    # if mock_logger: mock_logger.close_connection() # If mock logger had a connection

    logger.info("--- Quality Evaluator & Adaptive Control (Advanced with MQ) Example Finished ---")