from __future__ import annotations

from typing import List, Dict, Any
from .types import SuperWeightPrediction


class PredictionEvaluator:
    """Evaluates the quality of super-weight predictions."""
    
    def __init__(self) -> None:
        """Initialize the prediction evaluator."""
        pass
    
    def evaluate_predictions(self, predictions: List[SuperWeightPrediction]) -> Dict[str, Any]:
        """
        Evaluate a list of predictions and return metrics.
        
        Args:
            predictions: List of super-weight predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not predictions:
            return {
                "total_predictions": 0,
                "unique_layers": 0,
                "avg_layer": 0,
                "max_value": 0,
                "min_value": 0,
                "avg_value_abs": 0,
                "large_values_ratio": 0,  # Ratio of values with |value| > 10
            }
            
        # Calculate metrics
        total_predictions = len(predictions)
        layers = [p.layer for p in predictions]
        values = [p.value for p in predictions]
        abs_values = [abs(v) for v in values]
        
        unique_layers = len(set(layers))
        avg_layer = sum(layers) / len(layers)
        max_value = max(values)
        min_value = min(values)
        avg_value_abs = sum(abs_values) / len(abs_values)
        
        # Count large values (|value| > 10 is often characteristic of super-weights)
        large_values_count = sum(1 for v in abs_values if v > 10)
        large_values_ratio = large_values_count / total_predictions if total_predictions > 0 else 0
        
        return {
            "total_predictions": total_predictions,
            "unique_layers": unique_layers,
            "avg_layer": avg_layer,
            "max_value": max_value,
            "min_value": min_value,
            "avg_value_abs": avg_value_abs,
            "large_values_ratio": large_values_ratio,
        }
    
    def compare_predictions(self, pred1: List[SuperWeightPrediction], pred2: List[SuperWeightPrediction]) -> Dict[str, Any]:
        """
        Compare two sets of predictions.
        
        Args:
            pred1: First list of predictions
            pred2: Second list of predictions
            
        Returns:
            Dictionary of comparison metrics
        """
        eval1 = self.evaluate_predictions(pred1)
        eval2 = self.evaluate_predictions(pred2)
        
        comparison = {
            "pred1_metrics": eval1,
            "pred2_metrics": eval2,
        }
        
        # Calculate differences
        if eval1["total_predictions"] > 0 and eval2["total_predictions"] > 0:
            comparison["differences"] = {
                "total_predictions_diff": eval2["total_predictions"] - eval1["total_predictions"],
                "avg_layer_diff": eval2["avg_layer"] - eval1["avg_layer"],
                "avg_value_abs_diff": eval2["avg_value_abs"] - eval1["avg_value_abs"],
                "large_values_ratio_diff": eval2["large_values_ratio"] - eval1["large_values_ratio"],
            }
            
        return comparison