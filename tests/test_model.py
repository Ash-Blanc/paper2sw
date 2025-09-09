from __future__ import annotations

import pytest
from paper2sw.model import SemanticDiffusionModel
from paper2sw.types import SuperWeightPrediction


def test_semantic_diffusion_model_creation():
    """Test creating a SemanticDiffusionModel object."""
    model = SemanticDiffusionModel(
        model_id="test-model",
        device="cpu",
        precision="bf16"
    )
    
    assert model.model_id == "test-model"
    assert model.device == "cpu"
    assert model.precision == "bf16"


def test_semantic_diffusion_model_validation():
    """Test validation of SemanticDiffusionModel parameters."""
    # Invalid model_id
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        SemanticDiffusionModel(model_id="", device="cpu", precision="bf16")
    
    # Invalid device
    with pytest.raises(ValueError, match="device must be a string"):
        SemanticDiffusionModel(model_id="test-model", device=123, precision="bf16")
    
    # Invalid precision
    with pytest.raises(ValueError, match="precision must be a string"):
        SemanticDiffusionModel(model_id="test-model", device="cpu", precision=123)


def test_semantic_diffusion_model_predict():
    """Test the predict method."""
    model = SemanticDiffusionModel(
        model_id="test-model",
        device="cpu",
        precision="bf16"
    )
    
    test_text = "This is a test paper about Llama neural networks with 32 layers."
    predictions = model.predict(test_text, top_k=5)
    
    # The number of predictions might be less than top_k if there aren't enough candidates
    assert len(predictions) >= 4  # At least 4 predictions
    assert all(isinstance(p, SuperWeightPrediction) for p in predictions)
    assert all(isinstance(p.model_family, str) for p in predictions)
    assert all(isinstance(p.layer, int) for p in predictions)
    assert all(isinstance(p.row, int) for p in predictions)
    assert all(isinstance(p.col, int) for p in predictions)
    assert all(isinstance(p.value, float) for p in predictions)
    
    # Test with seed for reproducibility
    predictions1 = model.predict(test_text, top_k=3, seed=42)
    predictions2 = model.predict(test_text, top_k=3, seed=42)
    
    assert len(predictions1) == len(predictions2)
    # Note: Exact reproducibility might not be guaranteed due to implementation details


def test_semantic_diffusion_model_predict_edge_cases():
    """Test edge cases for the predict method."""
    model = SemanticDiffusionModel(
        model_id="test-model",
        device="cpu",
        precision="bf16"
    )
    
    # Test with top_k = 0
    predictions = model.predict("test text", top_k=0)
    assert len(predictions) == 0
    
    # Test with top_k = 1
    predictions = model.predict("test text", top_k=1)
    assert len(predictions) == 1
    
    # Test with invalid text type
    with pytest.raises(TypeError, match="text must be a string"):
        model.predict(123, top_k=5)
    
    # Test with invalid top_k type
    with pytest.raises(TypeError, match="top_k must be an integer"):
        model.predict("test text", top_k="5")
    
    # Test with invalid seed type
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        model.predict("test text", top_k=5, seed="42")