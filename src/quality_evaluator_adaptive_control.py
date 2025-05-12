# quality_evaluator_adaptive_control.py
# Description: Evaluates weld quality based on AI prediction and sensors,
#              and generates real-time adaptive control adjustments.

import config
# import numpy as np # Uncomment if direct numpy operations on sensor data are needed

# Re-define quality levels from evaluator.py for clarity or import if preferred
# For this example, copying is sufficient. In a real system, import or use a shared config.
QUALITY_CLASSES = {
    0: "Complete Fusion / Good",
    1: "Incomplete Fusion / Lack of Fusion",
    2: "Undercut",
    3: "Hot Tear / Crack"
    # Add more specific defect types or levels if needed
}

# --- Adaptive Control Rules (Placeholder) ---
# This dictionary defines example parameter adjustments based on predicted issues.
# In a real system, these rules would be highly complex, derived from
# process knowledge, experiments, or even another AI model/reinforcement learning policy.
# Format: { 'predicted_issue' : { 'parameter_to_adjust' : 'adjustment_value' } }
# Adjustment value can be absolute, relative ('+5', '-10'), or percentage.
# Using relative adjustments as example:
ADAPTIVE_CONTROL_RULES = {
    # If AI predicts Incomplete Fusion (often due to low heat/speed)
    "Incomplete Fusion / Lack of Fusion": {
        'speed': -10, # Decrease speed by 10 mm/min
        'current': +5 # Increase current by 5 Amps
    },
    # If AI predicts Undercut (often due to excessive voltage/speed or low current)
    "Undercut": {
        'voltage': -1, # Decrease voltage by 1 Volt
        'speed': -5  # Slight decrease in speed
    },
    # If AI predicts Hot Tear / Crack (complex, can relate to heat input, speed, material)
    "Hot Tear / Crack": {
        'speed': -15, # Significant speed decrease
        'current': -5 # Slight current decrease
        # More complex adjustments might be needed here, possibly involving pre/post-heat or gas composition
    },
    # Add rules based on sensor thresholds (e.g., 'excessive_spatter_detected')
    # "Excessive Spatter": {
    #     'voltage': +0.5 # Slight voltage increase might help reduce spatter
    # }
    # Add rules for 'Good' weld if optimizing for speed/efficiency
    # "Complete Fusion / Good": {
    #    'speed': +5 # Slightly increase speed if quality is consistently good
    # }
}

# Adjustment limits to prevent runaway changes (Example: max +/- 10% relative change per step)
MAX_RELATIVE_ADJUSTMENT_PERCENT = 5 # Max 5% change per adjustment cycle for a parameter
MIN_PARAM_VALUES = {param: rng[0] for param, rng in config.PARAM_RANGES.items()}
MAX_PARAM_VALUES = {param: rng[1] for param, rng in config.PARAM_RANGES.items()}
MIN_ADJ_VALUES = {param: rng[0] for param, rng in config.ADJUSTMENT_PARAMS.items()}
MAX_ADJ_VALUES = {param: rng[1] for param, rng in config.ADJUSTMENT_PARAMS.items()}


class QualityEvaluatorAdaptiveControl:
    """
    Evaluates weld quality and determines adaptive control actions.
    Combines AI prediction, sensor data rules, and generates parameter adjustments.
    """

    def __init__(self):
        print("Quality Evaluator and Adaptive Control module initialized.")
        pass # No complex state needed initially

    def evaluate_quality(self, ai_prediction, real_time_sensor_data=None, physics_prediction=None):
        """
        Evaluates the weld quality based on multiple data sources.

        Args:
            ai_prediction (dict): Output from AIInferenceEngine.process_sensor_data.
                                  (e.g., {'predicted_class': 1, 'confidence': 0.7, ...} or {'predicted_score': 0.8})
            real_time_sensor_data (dict, optional): Dictionary of real-time sensor readings.
                                                   (e.g., {'temperature': 450, 'arc_stability_index': 0.6})
            physics_prediction (dict, optional): Output from physics simulation (if available).
                                                (e.g., {'predicted_bead_shape': 'Good', 'quality_score': 0.9})


        Returns:
            dict: Comprehensive evaluation results including quality level, scores,
                  detected issues, and a summary status (e.g., 'Good', 'Warning', 'Critical').
        """
        evaluation_results = {
            'ai_evaluation': None,
            'sensor_rule_evaluation': {},
            'physics_evaluation': None, # If used
            'combined_status': 'Unknown', # Good, Warning, Critical
            'detected_issues': []
        }

        # --- 1. Evaluate based on AI Prediction (Reuse logic from evaluator.py) ---
        if ai_prediction:
            ai_eval = {}
            if 'predicted_class' in ai_prediction:
                pred_class = ai_prediction['predicted_class']
                ai_eval['quality_level'] = QUALITY_CLASSES.get(pred_class, f"Unknown Class ({pred_class})")
                ai_eval['confidence'] = ai_prediction.get('confidence')
                # Derive a numeric score if needed, similar to evaluator.py
                ai_eval['numeric_score'] = 1.0 - ai_prediction.get('confidence', 0.5) if pred_class != 0 else ai_prediction.get('confidence', 0.5) # Example
                evaluation_results['ai_evaluation'] = ai_eval
                evaluation_results['detected_issues'].append(ai_eval['quality_level']) # Add predicted defect as an issue

            elif 'predicted_score' in ai_prediction: # Regression case
                score = ai_prediction['predicted_score']
                ai_eval['numeric_score'] = score
                if score > 0.85: ai_eval['quality_level'] = "Good"
                elif score > 0.6: ai_eval['quality_level'] = "Acceptable"
                else: ai_eval['quality_level'] = "Poor / Defect Likely"
                evaluation_results['ai_evaluation'] = ai_eval
                if score <= 0.85: evaluation_results['detected_issues'].append(ai_eval['quality_level'])


        # --- 2. Evaluate based on Real-time Sensor Data (Rule-based - Placeholder Rules) ---
        if real_time_sensor_data:
            # Example Rule 1: Check for unusually high temperature
            if 'temperature' in real_time_sensor_data and real_time_sensor_data['temperature'] > 400: # Example threshold
                evaluation_results['sensor_rule_evaluation']['high_temperature'] = True
                evaluation_results['detected_issues'].append("High Temperature Detected")
                evaluation_results['combined_status'] = 'Warning' # Higher priority status

            # Example Rule 2: Check for low arc stability index (if available)
            if 'arc_stability_index' in real_time_sensor_data and real_time_sensor_data['arc_stability_index'] < 0.5: # Example threshold
                 evaluation_results['sensor_rule_evaluation']['low_arc_stability'] = True
                 evaluation_results['detected_issues'].append("Low Arc Stability Detected")
                 evaluation_results['combined_status'] = 'Critical' # Highest priority status (might indicate imminent problem)

            # Add more sensor-based rules as needed...


        # --- 3. Evaluate based on Physics Simulation (Optional) ---
        # Integrate physics prediction results if available, possibly for validation
        if physics_prediction:
            # Example: Check if physics prediction contradicts AI or is also poor
            evaluation_results['physics_evaluation'] = physics_prediction
            if physics_prediction.get('quality_score') is not None and physics_prediction['quality_score'] < 0.6:
                 evaluation_results['detected_issues'].append("Physics Simulation Predicts Poor Quality")
                 # Potentially upgrade status based on physics prediction confirmation


        # --- 4. Determine Combined Status ---
        # Default to Good if no issues detected or status not already set by critical rules
        if evaluation_results['combined_status'] == 'Unknown':
             if len(evaluation_results['detected_issues']) == 0 or all("Good" in issue for issue in evaluation_results['detected_issues']):
                  evaluation_results['combined_status'] = 'Good'
             else:
                  # If any non-Good issues were detected by AI or rules, mark as Warning/Critical
                  evaluation_results['combined_status'] = 'Warning' # Default to Warning if not Critical by rule


        print(f"Quality Evaluation Result (Status: {evaluation_results['combined_status']}): {evaluation_results}")
        return evaluation_results

    def generate_adaptive_adjustments(self, evaluation_results, current_welding_params):
        """
        Generates adaptive control parameter adjustments based on evaluation results.

        Args:
            evaluation_results (dict): Output from evaluate_quality.
            current_welding_params (dict): Dictionary of current welding parameters
                                          (e.g., {'current': 160, 'voltage': 23, 'speed': 250}).
                                          Includes parameters from config.PARAM_RANGES and ADJUSTMENT_PARAMS.

        Returns:
            dict or None: Dictionary of suggested parameter adjustments
                          (e.g., {'current': 165.0, 'speed': 240.0})
                          or None if no adjustments are needed or possible.
                          The values are the *target* absolute values.
        """
        suggested_adjustments = {}
        status = evaluation_results.get('combined_status')
        detected_issues = evaluation_results.get('detected_issues', [])

        if status == 'Good':
            # Consider minor optimization adjustments if consistently good
            # For this example, no adjustments for 'Good' status
            print("Status is Good. No adaptive adjustments suggested.")
            return None # No adjustments needed

        elif status == 'Warning' or status == 'Critical':
            print(f"Status is {status}. Generating adaptive adjustments...")
            applied_rules = set()

            # Prioritize adjustments based on detected issues (e.g., specific predicted defects)
            for issue in detected_issues:
                if issue in ADAPTIVE_CONTROL_RULES:
                    rules_to_apply = ADAPTIVE_CONTROL_RULES[issue]
                    print(f" Applying rule for issue: {issue}")
                    for param, adjustment in rules_to_apply.items():
                        # Ensure we only apply one rule per parameter per cycle if rules conflict
                        if param not in suggested_adjustments:
                            if param in current_welding_params:
                                current_value = current_welding_params[param]
                                # Calculate target value based on relative adjustment (example logic)
                                if isinstance(adjustment, (int, float)): # Simple numeric adjustment
                                    target_value = current_value + adjustment
                                elif isinstance(adjustment, str) and adjustment.startswith('+'):
                                     target_value = current_value + float(adjustment[1:])
                                elif isinstance(adjustment, str) and adjustment.startswith('-'):
                                     target_value = current_value - float(adjustment[1:])
                                # Add percentage or other logic if needed
                                else:
                                     print(f"  Warning: Unrecognized adjustment format for {param}: {adjustment}")
                                     continue

                                # --- Apply Limits ---
                                # Prevent exceeding parameter ranges defined in config
                                min_limit = MIN_PARAM_VALUES.get(param) or MIN_ADJ_VALUES.get(param)
                                max_limit = MAX_PARAM_VALUES.get(param) or MAX_ADJ_VALUES.get(param)

                                if min_limit is not None and target_value < min_limit:
                                     target_value = min_limit
                                     print(f"  Limited {param} adjustment to min limit: {min_limit}")
                                if max_limit is not None and target_value > max_limit:
                                     target_value = max_limit
                                     print(f"  Limited {param} adjustment to max limit: {max_limit}")

                                # Apply max relative change per step (e.g., 5%)
                                max_delta = current_value * (MAX_RELATIVE_ADJUSTMENT_PERCENT / 100.0)
                                if abs(target_value - current_value) > max_delta:
                                    sign = 1 if target_value > current_value else -1
                                    target_value = current_value + sign * max_delta
                                    # Re-apply hard limits after relative limit
                                    if min_limit is not None: target_value = max(target_value, min_limit)
                                    if max_limit is not None: target_value = min(target_value, max_limit)
                                    print(f"  Limited {param} adjustment by max relative change ({MAX_RELATIVE_ADJUSTMENT_PERCENT}%). New target: {target_value:.2f}")


                                suggested_adjustments[param] = round(target_value, 2) # Round for practical application
                                applied_rules.add(issue)
                            else:
                                print(f"  Warning: Parameter '{param}' to adjust not found in current_welding_params.")

            if not suggested_adjustments:
                 print("No specific rules matched detected issues or parameters missing. No adjustments suggested.")
                 return None

            print(f"Suggested Adjustments: {suggested_adjustments}")
            return suggested_adjustments

        else:
            print("Unknown status. No adaptive adjustments suggested.")
            return None # Unknown status, no adjustments

    # This module would NOT send commands directly, but return them to a manager
    # (e.g., welding_process_manager) which then sends to robot_control_interface.
    # Example signature for the manager:
    # process_manager.apply_welding_adjustments(suggested_adjustments)


# Example Usage (requires dummy inputs mimicking other modules)
if __name__ == '__main__':
    print("--- Quality Evaluator and Adaptive Control Example ---")

    # --- Create Dummy Config (for example purposes only) ---
    # Reuse the DummyConfig from ai_inference_engine or load real config
    class DummyConfig:
        OUTPUT_CLASSES = 4
        PARAM_RANGES = {
            'current': (80, 200),
            'voltage': (15, 30),
            'speed': (100, 500)
        }
        ADJUSTMENT_PARAMS = {
            'torch_angle': (0, 45), # Degrees
            'ctwd': (10, 20)
        }
        # Add other relevant configs

    config = DummyConfig() # Use dummy config

    # --- Initialize the Evaluator/Controller ---
    evaluator_controller = QualityEvaluatorAdaptiveControl()

    # --- Example Inputs (Mimicking outputs from other modules) ---

    # Scenario 1: AI predicts Good Weld
    print("\n--- Scenario 1: AI Predicts Good Weld ---")
    dummy_ai_prediction_good = {'predicted_class': 0, 'probabilities': [0.9, 0.05, 0.03, 0.02], 'confidence': 0.9}
    dummy_sensor_data_good = {'temperature': 350, 'arc_stability_index': 0.8}
    current_params_good = {'current': 160, 'voltage': 23, 'speed': 250, 'torch_angle': 10, 'ctwd': 15} # Include adj params

    eval_results_good = evaluator_controller.evaluate_quality(
        ai_prediction=dummy_ai_prediction_good,
        real_time_sensor_data=dummy_sensor_data_good,
        current_welding_params=current_params_good # Passing current params here is only for adjustment logic, not evaluation
    )
    adjustments_good = evaluator_controller.generate_adaptive_adjustments(
        evaluation_results=eval_results_good,
        current_welding_params=current_params_good
    )
    print("Suggested Adjustments 1:", adjustments_good) # Should be None

    # Scenario 2: AI predicts Undercut
    print("\n--- Scenario 2: AI Predicts Undercut ---")
    dummy_ai_prediction_undercut = {'predicted_class': 2, 'probabilities': [0.05, 0.1, 0.75, 0.1], 'confidence': 0.75}
    dummy_sensor_data_undercut = {'temperature': 420, 'arc_stability_index': 0.7} # High temp detected by rule
    current_params_undercut = {'current': 170, 'voltage': 25, 'speed': 300, 'torch_angle': 8, 'ctwd': 16}

    eval_results_undercut = evaluator_controller.evaluate_quality(
        ai_prediction=dummy_ai_prediction_undercut,
        real_time_sensor_data=dummy_sensor_data_undercut,
        current_welding_params=current_params_undercut # Passing current params here is only for adjustment logic, not evaluation
    )
    adjustments_undercut = evaluator_controller.generate_adaptive_adjustments(
        evaluation_results=eval_results_undercut,
        current_welding_params=current_params_undercut
    )
    print("Suggested Adjustments 2:", adjustments_undercut) # Should suggest voltage/speed decrease and maybe current increase rule for Lack of Fusion if applicable

    # Scenario 3: AI predicts Lack of Fusion (Regression Score)
    print("\n--- Scenario 3: AI Predicts Poor Score (Regression) ---")
    dummy_ai_prediction_poor_score = {'predicted_score': 0.55}
    dummy_sensor_data_stable = {'temperature': 380, 'arc_stability_index': 0.85}
    current_params_poor_score = {'current': 150, 'voltage': 22, 'speed': 350, 'torch_angle': 12, 'ctwd': 14}

    eval_results_poor_score = evaluator_controller.evaluate_quality(
        ai_prediction=dummy_ai_prediction_poor_score,
        real_time_sensor_data=dummy_sensor_data_stable,
        current_welding_params=current_params_poor_score
    )
    adjustments_poor_score = evaluator_controller.generate_adaptive_adjustments(
        evaluation_results=eval_results_poor_score,
        current_welding_params=current_params_poor_score
    )
    print("Suggested Adjustments 3:", adjustments_poor_score) # Should suggest adjustments based on 'Poor / Defect Likely' if rules exist

    # Scenario 4: Sensor rule triggers Critical status
    print("\n--- Scenario 4: Sensor Rule Triggers Critical Status ---")
    dummy_ai_prediction_ok = {'predicted_class': 0, 'confidence': 0.8} # AI thinks it's okay
    dummy_sensor_data_critical = {'temperature': 390, 'arc_stability_index': 0.3} # Very low stability
    current_params_critical = {'current': 180, 'voltage': 24, 'speed': 260, 'torch_angle': 9, 'ctwd': 17}

    eval_results_critical = evaluator_controller.evaluate_quality(
        ai_prediction=dummy_ai_prediction_ok,
        real_time_sensor_data=dummy_sensor_data_critical,
        current_welding_params=current_params_critical
    )
    adjustments_critical = evaluator_controller.generate_adaptive_adjustments(
        evaluation_results=eval_results_critical,
        current_welding_params=current_params_critical
    )
    print("Suggested Adjustments 4:", adjustments_critical) # Should suggest adjustments based on detected issue

    print("\n--- Example Finished ---")