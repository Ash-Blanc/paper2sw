from __future__ import annotations

import pytest
from pathlib import Path
from paper2sw.predictor import Predictor
from paper2sw.types import SuperWeightPrediction


def test_predictor_creation():
    """Test creating a Predictor object."""
    predictor = Predictor.from_pretrained()
    
    assert predictor.model_id == "paper2sw/paper2sw-diff-base"
    assert predictor.device == "cpu"
    assert predictor.precision == "bf16"
    assert predictor.backend == "dummy"
    assert predictor.selection_keep_ratio == 1.0


def test_predictor_with_custom_settings():
    """Test creating a Predictor with custom settings."""
    predictor = Predictor.from_pretrained(
        model_id="test-model",
        device="cuda:0",
        precision="fp16",
        enable_cache=False,
        selection_keep_ratio=0.5,
        backend="test-backend"
    )
    
    assert predictor.model_id == "test-model"
    assert predictor.device == "cuda:0"
    assert predictor.precision == "fp16"
    assert predictor.backend == "test-backend"
    assert predictor.selection_keep_ratio == 0.5
    assert predictor.cache.enabled == False


def test_predictor_from_config():
    """Test creating a Predictor from a configuration dictionary."""
    config = {
        "model_id": "config-model",
        "device": "cuda:1",
        "precision": "fp32",
        "enable_cache": False,
        "selection_keep_ratio": 0.3,
        "backend": "config-backend"
    }
    
    predictor = Predictor.from_config(config)
    
    assert predictor.model_id == "config-model"
    assert predictor.device == "cuda:1"
    assert predictor.precision == "fp32"
    assert predictor.backend == "config-backend"
    assert predictor.selection_keep_ratio == 0.3
    assert predictor.cache.enabled == False


def test_predictor_validation():
    """Test validation of Predictor parameters."""
    # Invalid model_id
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        Predictor.from_pretrained(model_id="")
    
    # Invalid device
    with pytest.raises(ValueError, match="device must be a string"):
        Predictor.from_pretrained(device=123)
    
    # Invalid precision
    with pytest.raises(ValueError, match="precision must be a string"):
        Predictor.from_pretrained(precision=123)
    
    # Invalid enable_cache
    with pytest.raises(ValueError, match="enable_cache must be a boolean"):
        Predictor.from_pretrained(enable_cache="true")
    
    # Invalid selection_keep_ratio
    with pytest.raises(ValueError, match="selection_keep_ratio must be a float between 0.0 and 1.0"):
        Predictor.from_pretrained(selection_keep_ratio=1.5)
    
    # Invalid backend
    with pytest.raises(ValueError, match="backend must be a string"):
        Predictor.from_pretrained(backend=123)


def test_predictor_predict():
    """Test the predict method with a simple input."""
    predictor = Predictor.from_pretrained()
    
    # Use a simple text for testing
    test_text = "This is a test paper about Llama models."
    
    # Write test text to a temporary file
    test_file = Path("test_paper.txt")
    test_file.write_text(test_text)
    
    try:
        predictions = predictor.predict(str(test_file), top_k=3)
        
        assert len(predictions) == 3
        assert all(isinstance(p, SuperWeightPrediction) for p in predictions)
        assert all(p.model_family == "Llama-7B" for p in predictions)  # Should be detected from text
    finally:
        # Clean up test file
        test_file.unlink()


def test_predictor_predict_batch():
    """Test the predict_batch method."""
    predictor = Predictor.from_pretrained()
    
    # Create test files
    test_text1 = "This is a test paper about Llama models."
    test_text2 = "This is another test paper about Mistral models."
    
    test_file1 = Path("test_paper1.txt")
    test_file2 = Path("test_paper2.txt")
    
    test_file1.write_text(test_text1)
    test_file2.write_text(test_text2)
    
    try:
        results = predictor.predict_batch([str(test_file1), str(test_file2)], top_k=2)
        
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2
        assert all(isinstance(p, SuperWeightPrediction) for preds in results for p in preds)
    finally:
        # Clean up test files
        test_file1.unlink()
        test_file2.unlink()