from __future__ import annotations

import pytest
from paper2sw.evaluator import PredictionEvaluator
from paper2sw.types import SuperWeightPrediction


def test_prediction_evaluator_creation():
    """Test creating a PredictionEvaluator object."""
    evaluator = PredictionEvaluator()
    assert evaluator is not None


def test_evaluate_empty_predictions():
    """Test evaluating an empty list of predictions."""
    evaluator = PredictionEvaluator()
    metrics = evaluator.evaluate_predictions([])
    
    assert metrics["total_predictions"] == 0
    assert metrics["unique_layers"] == 0
    assert metrics["avg_layer"] == 0
    assert metrics["max_value"] == 0
    assert metrics["min_value"] == 0
    assert metrics["avg_value_abs"] == 0
    assert metrics["large_values_ratio"] == 0


def test_evaluate_predictions():
    """Test evaluating a list of predictions."""
    evaluator = PredictionEvaluator()
    
    predictions = [
        SuperWeightPrediction(model_family="Test-Model", layer=0, row=100, col=200, value=15.5),
        SuperWeightPrediction(model_family="Test-Model", layer=1, row=150, col=250, value=-12.3),
        SuperWeightPrediction(model_family="Test-Model", layer=2, row=300, col=400, value=8.7),
        SuperWeightPrediction(model_family="Test-Model", layer=0, row=120, col=220, value=-18.9),
    ]
    
    metrics = evaluator.evaluate_predictions(predictions)
    
    assert metrics["total_predictions"] == 4
    assert metrics["unique_layers"] == 3  # Layers 0, 1, 2
    assert metrics["avg_layer"] == 0.75  # (0+1+2+0)/4
    assert metrics["max_value"] == 15.5
    assert metrics["min_value"] == -18.9
    assert abs(metrics["avg_value_abs"] - 13.85) < 0.01  # (15.5+12.3+8.7+18.9)/4 = 13.85
    assert abs(metrics["large_values_ratio"] - 0.75) < 0.01  # 3 out of 4 values have |value| > 10


def test_compare_predictions():
    """Test comparing two sets of predictions."""
    evaluator = PredictionEvaluator()
    
    pred1 = [
        SuperWeightPrediction(model_family="Test-Model", layer=0, row=100, col=200, value=15.5),
        SuperWeightPrediction(model_family="Test-Model", layer=1, row=150, col=250, value=-12.3),
    ]
    
    pred2 = [
        SuperWeightPrediction(model_family="Test-Model", layer=0, row=100, col=200, value=18.5),
        SuperWeightPrediction(model_family="Test-Model", layer=1, row=150, col=250, value=-15.3),
        SuperWeightPrediction(model_family="Test-Model", layer=2, row=300, col=400, value=9.7),
    ]
    
    comparison = evaluator.compare_predictions(pred1, pred2)
    
    assert "pred1_metrics" in comparison
    assert "pred2_metrics" in comparison
    assert "differences" in comparison
    
    # Check that pred2 has more predictions
    assert comparison["differences"]["total_predictions_diff"] == 1  # 3 - 2 = 1
    
    # Check that pred2 has a higher average absolute value
    assert comparison["differences"]["avg_value_abs_diff"] > 0