from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from paper2sw.api import predict_super_weights, predict_super_weights_batch, save_predictions_csv
from paper2sw.types import SuperWeightPrediction


def test_predict_super_weights():
    """Test the predict_super_weights function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_content = "This is a test paper about Llama models."
        test_file.write_text(test_content)
        
        predictions = predict_super_weights(str(test_file), top_k=3)
        
        assert len(predictions) == 3
        assert all(isinstance(p, SuperWeightPrediction) for p in predictions)
        assert all(isinstance(p.model_family, str) for p in predictions)
        assert all(isinstance(p.layer, int) for p in predictions)
        assert all(isinstance(p.row, int) for p in predictions)
        assert all(isinstance(p.col, int) for p in predictions)
        assert all(isinstance(p.value, float) for p in predictions)


def test_predict_super_weights_validation():
    """Test validation in predict_super_weights function."""
    # Test with invalid paper type
    with pytest.raises(TypeError, match="paper must be a string or Path"):
        predict_super_weights(123, top_k=5)
    
    # Test with invalid top_k
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_super_weights("./test.txt", top_k=0)
    
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_super_weights("./test.txt", top_k=-1)


def test_predict_super_weights_batch():
    """Test the predict_super_weights_batch function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file1 = Path(tmpdir) / "test1.txt"
        test_file2 = Path(tmpdir) / "test2.txt"
        
        test_file1.write_text("This is a test paper about Llama models.")
        test_file2.write_text("This is a test paper about Mistral models.")
        
        results = predict_super_weights_batch([str(test_file1), str(test_file2)], top_k=2)
        
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2
        assert all(isinstance(p, SuperWeightPrediction) for preds in results for p in preds)


def test_predict_super_weights_batch_validation():
    """Test validation in predict_super_weights_batch function."""
    # Test with invalid papers type
    with pytest.raises(TypeError, match="papers must be iterable"):
        predict_super_weights_batch("not_iterable", top_k=5)
    
    # Test with invalid top_k
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_super_weights_batch(["./test.txt"], top_k=0)
    
    # Test with invalid keep_ratio
    with pytest.raises(ValueError, match="keep_ratio must be a float between 0.0 and 1.0"):
        predict_super_weights_batch(["./test.txt"], top_k=5, keep_ratio=1.5)


def test_save_predictions_csv():
    """Test the save_predictions_csv function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.csv"
        
        predictions = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            )
        ]
        
        save_predictions_csv(predictions, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Read back and verify content
        content = output_file.read_text()
        assert "Llama-7B,2,3968,7003,-17.328" in content


def test_save_predictions_csv_validation():
    """Test validation in save_predictions_csv function."""
    # Test with invalid predictions type
    with pytest.raises(TypeError, match="predictions must be a list"):
        save_predictions_csv("not_a_list", "./output.csv")