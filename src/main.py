# main.py
# Description: Main script to run the welding prediction and evaluation process.

import time
import numpy as np
import pandas as pd # For reading potential input files

import config # Import configuration settings
from predictor import Predictor
from evaluator import WeldEvaluator
# from data_handler import load_data # Optional: for getting test data

def run_single_prediction(predictor, evaluator, input_params, prefer='ai'):
    """
    Runs a single prediction cycle for given input parameters.

    Args:
        predictor (Predictor): The predictor object.
        evaluator (WeldEvaluator): The evaluator object.
        input_params (dict): Dictionary containing input parameters for both
                             AI (sensor values) and physics (welding params).
                             Example: {'current': 150, 'voltage': 22, 'speed': 300, 'temp': 350,
                                      'torch_angle': 10, 'ctwd': 15, 'gas_flow': 20}
        prefer (str): 'ai' or 'physics' - preferred prediction method.

    Returns:
        dict: Combined results including input, prediction, and evaluation.
    """
    print("-" * 50)
    print(f"Running prediction for input: {input_params}")
    print(f"Preferred method: {prefer}")

    # Prepare inputs for AI and Physics based on the combined input_params
    # This assumes input_params contains all necessary keys
    # You might need more sophisticated logic to map inputs
    sensor_keys = ['current', 'voltage', 'speed', 'temperature'] # Match AI model input
    ai_sensor_input = np.array([[input_params.get(k, 0) for k in sensor_keys]]) # Default missing to 0
    # Note: Scaling might be needed here based on how Predictor is implemented!

    # Physics params can be the whole dict or a subset
    physics_welding_params = input_params.copy()

    # --- Perform Prediction ---
    start_time = time.time()
    method_used, prediction_result = predictor.predict(
        sensor_input=ai_sensor_input,
        # image_input=None, # Add image input if available/needed
        welding_params=physics_welding_params,
        prefer=prefer
    )
    end_time = time.time()
    prediction_time = end_time - start_time

    # --- Perform Evaluation ---
    if prediction_result:
        evaluation_result = evaluator.evaluate(method_used, prediction_result)
        print(f"\nPrediction Method Used: {method_used}")
        print(f"Prediction Result: {prediction_result}")
        print(f"Evaluation Result: {evaluation_result}")
        print(f"Prediction Time: {prediction_time:.4f} seconds")

        combined_output = {
            'input_parameters': input_params,
            'prediction_method': method_used,
            'prediction_output': prediction_result,
            'evaluation': evaluation_result,
            'prediction_time_sec': prediction_time
        }
    else:
        print("\nPrediction Failed.")
        print(f"Time taken: {prediction_time:.4f} seconds")
        combined_output = {
            'input_parameters': input_params,
            'prediction_method': None,
            'prediction_output': None,
            'evaluation': {'quality_level': 'Failed Prediction'},
            'prediction_time_sec': prediction_time
        }

    print("-" * 50)
    return combined_output


if __name__ == "__main__":
    print("--- Welding Bead Prediction System ---")

    # --- Initialize Components ---
    # Choose whether to enable AI and/or Physics prediction
    # Set use_ai=False if you haven't trained the model (no .pth file)
    # Set use_physics=False if you don't have the UE simulator running
    predictor = Predictor(use_ai=True, use_physics=False) # Example: Use AI only
    evaluator = WeldEvaluator()

    # --- Define Input Scenarios ---
    # These would typically come from a UI, a file, or live sensors
    test_scenario_1 = {
        'current': 165, 'voltage': 24, 'speed': 280, 'temperature': 400, # AI inputs
        'torch_angle': 8, 'ctwd': 18, 'gas_flow': 18 # Physics inputs (+ above)
    }
    test_scenario_2 = {
        'current': 100, 'voltage': 18, 'speed': 450, 'temperature': 250, # AI inputs
        'torch_angle': 15, 'ctwd': 12, 'gas_flow': 22 # Physics inputs (+ above)
    }
    test_scenario_3 = { # Example with potentially problematic parameters
        'current': 190, 'voltage': 28, 'speed': 200, 'temperature': 500, # AI inputs
        'torch_angle': 2, 'ctwd': 20, 'gas_flow': 15 # Physics inputs (+ above)
    }

    # --- Run Predictions ---
    results = []
    print("\nStarting Prediction Runs...")

    # Run scenario 1, preferring AI
    result1 = run_single_prediction(predictor, evaluator, test_scenario_1, prefer='ai')
    results.append(result1)

    # Run scenario 2, preferring physics (if enabled)
    # If physics is disabled in Predictor init, it should fallback or fail gracefully
    result2 = run_single_prediction(predictor, evaluator, test_scenario_2, prefer='physics')
    results.append(result2)

    # Run scenario 3, default preference (AI if available)
    result3 = run_single_prediction(predictor, evaluator, test_scenario_3)
    results.append(result3)

    # --- Post-Processing / Reporting ---
    print("\n--- Summary of Predictions ---")
    for i, res in enumerate(results):
        print(f"\nRun {i+1}:")
        print(f"  Input: {res['input_parameters']}")
        print(f"  Method: {res['prediction_method']}")
        # print(f"  Prediction: {res['prediction_output']}") # Can be verbose
        print(f"  Evaluation: {res['evaluation']}")
        print(f"  Time: {res['prediction_time_sec']:.4f}s")

    # Optional: Save results to a file
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv('prediction_log.csv', index=False)
        print("\nPrediction log saved to prediction_log.csv")
    except Exception as e:
        print(f"\nCould not save results to CSV: {e}")


    # --- Cleanup ---
    # Disconnect physics simulator if it was used and connected
    if predictor.physics_simulator and predictor.physics_simulator.is_connected:
        predictor.physics_simulator.disconnect()

    print("\n--- System Finished ---")
    