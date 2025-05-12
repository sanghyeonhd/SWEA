# evaluator.py
# Description: Evaluates the quality of the weld based on prediction results.

import config

# Define quality levels based on the image description
QUALITY_CLASSES = {
    0: "Complete Fusion / Good",
    1: "Incomplete Fusion / Lack of Fusion",
    2: "Undercut",
    3: "Hot Tear / Crack"
    # Add more specific defect types or levels if needed
}

class WeldEvaluator:
    """Evaluates weld quality based on AI or physics predictions."""

    def __init__(self):
        pass # No specific initialization needed for this simple version

    def evaluate_from_ai(self, ai_prediction):
        """
        Evaluates quality based on AI model output.

        Args:
            ai_prediction (dict): Output from Predictor.predict_with_ai
                                  (e.g., {'predicted_class': 1, 'probabilities': [...]})
                                  or {'predicted_score': 0.8}.

        Returns:
            dict: Evaluation results (e.g., {'quality_level': 'Incomplete Fusion', 'confidence': 0.7, 'numeric_score': 0.7})
        """
        if ai_prediction is None:
            return {'quality_level': 'Unknown', 'message': 'No AI prediction provided.'}

        evaluation = {}

        if 'predicted_class' in ai_prediction:
            pred_class = ai_prediction['predicted_class']
            evaluation['quality_level'] = QUALITY_CLASSES.get(pred_class, f"Unknown Class ({pred_class})")

            if 'probabilities' in ai_prediction:
                try:
                    confidence = ai_prediction['probabilities'][pred_class]
                    evaluation['confidence'] = round(float(confidence), 4)
                except (IndexError, TypeError):
                     evaluation['confidence'] = None # Handle cases where probabilities might be missing or malformed

            # You could derive a numeric score from the class/confidence if needed
            # Example simple mapping: Good=1.0, Others=0.5 - (confidence * 0.5)
            if pred_class == 0: # Assuming 0 is 'Good'
                evaluation['numeric_score'] = evaluation.get('confidence', 1.0) # Score = confidence if good
            else:
                evaluation['numeric_score'] = 1.0 - evaluation.get('confidence', 0.5) # Lower score for defects

        elif 'predicted_score' in ai_prediction: # Regression case
             score = ai_prediction['predicted_score']
             evaluation['numeric_score'] = round(float(score), 4)
             # Define quality levels based on score thresholds
             if score > 0.85:
                 evaluation['quality_level'] = "Good"
             elif score > 0.6:
                  evaluation['quality_level'] = "Acceptable"
             else:
                  evaluation['quality_level'] = "Poor / Defect Likely"
        else:
             evaluation['quality_level'] = 'Unknown'
             evaluation['message'] = 'AI prediction format not recognized.'


        return evaluation


    def evaluate_from_physics(self, physics_prediction):
        """
        Evaluates quality based on physics simulation output.

        Args:
            physics_prediction (dict): Output from Predictor.predict_with_physics
                                       (e.g., {'predicted_bead_shape': 'Good Bead', 'quality_score': 0.9})

        Returns:
            dict: Evaluation results (e.g., {'quality_level': 'Good Bead', 'numeric_score': 0.9})
        """
        if physics_prediction is None:
            return {'quality_level': 'Unknown', 'message': 'No physics prediction provided.'}

        evaluation = {}
        shape_desc = physics_prediction.get('predicted_bead_shape', 'Unknown Shape')
        score = physics_prediction.get('quality_score', None)

        # Assume the shape description directly maps to quality level for simplicity
        evaluation['quality_level'] = shape_desc

        if score is not None:
             try:
                 evaluation['numeric_score'] = round(float(score), 4)
             except (ValueError, TypeError):
                 evaluation['numeric_score'] = None # Handle non-numeric scores
        else:
             # Derive score from shape if needed (e.g., simple mapping)
             if "Good" in shape_desc: evaluation['numeric_score'] = 0.9
             elif "Undercut" in shape_desc: evaluation['numeric_score'] = 0.4
             elif "Fusion" in shape_desc: evaluation['numeric_score'] = 0.3
             else: evaluation['numeric_score'] = 0.5 # Default for unknown shapes


        return evaluation

    def evaluate(self, prediction_method, prediction_result):
        """
        Evaluates the result from either AI or physics prediction.

        Args:
            prediction_method (str): 'ai' or 'physics'.
            prediction_result (dict): The result dictionary from the predictor.

        Returns:
            dict: The evaluation dictionary.
        """
        if prediction_method == 'ai':
            return self.evaluate_from_ai(prediction_result)
        elif prediction_method == 'physics':
            return self.evaluate_from_physics(prediction_result)
        else:
            return {'quality_level': 'Unknown', 'message': f'Invalid prediction method: {prediction_method}'}


if __name__ == '__main__':
    evaluator = WeldEvaluator()

    # --- Example AI Prediction (Classification) ---
    ai_pred_class = {'predicted_class': 2, 'probabilities': [0.05, 0.15, 0.7, 0.1]}
    evaluation_ai_class = evaluator.evaluate('ai', ai_pred_class)
    print("--- AI Classification Evaluation ---")
    print(f"Input: {ai_pred_class}")
    print(f"Evaluation: {evaluation_ai_class}")

    # --- Example AI Prediction (Regression) ---
    ai_pred_reg = {'predicted_score': 0.65}
    evaluation_ai_reg = evaluator.evaluate('ai', ai_pred_reg)
    print("\n--- AI Regression Evaluation ---")
    print(f"Input: {ai_pred_reg}")
    print(f"Evaluation: {evaluation_ai_reg}")

    # --- Example Physics Prediction ---
    physics_pred = {'predicted_bead_shape': 'Lack of Fusion Zone', 'quality_score': 0.35}
    evaluation_physics = evaluator.evaluate('physics', physics_pred)
    print("\n--- Physics Simulation Evaluation ---")
    print(f"Input: {physics_pred}")
    print(f"Evaluation: {evaluation_physics}")

    # --- Example Failed Prediction ---
    failed_pred = None
    evaluation_failed = evaluator.evaluate('ai', failed_pred)
    print("\n--- Failed Prediction Evaluation ---")
    print(f"Input: {failed_pred}")
    print(f"Evaluation: {evaluation_failed}")
    