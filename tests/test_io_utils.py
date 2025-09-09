from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from paper2sw.io_utils import is_url, read_text_from_source, write_jsonl, write_csv
from paper2sw.types import SuperWeightPrediction


def test_is_url():
    \"\"\"Test the is_url function.\"\"\"
    # Test valid URLs
    assert is_url("http://example.com") == True
    assert is_url("https://example.com") == True
    assert is_url("https://arxiv.org/abs/2411.07191") == True
    
    # Test invalid URLs
    assert is_url("not_a_url") == False
    assert is_url("/path/to/file") == False
    assert is_url("./relative/path") == False
    assert is_url("") == False


def test_read_text_from_source_file():
    \"\"\"Test reading text from a file.\"\"\"
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_content = "This is a test file content.
With multiple lines."
        test_file.write_text(test_content)
        
        content = read_text_from_source(str(test_file))
        assert content == test_content


def test_read_text_from_source_nonexistent_file():
    \"\"\"Test reading text from a nonexistent file.\"\"\"
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_file = Path(tmpdir) / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            read_text_from_source(str(nonexistent_file))


def test_write_jsonl():
    \"\"\"Test writing predictions to JSONL format.\"\"\"
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.jsonl"
        
        predictions = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            ),
            SuperWeightPrediction(
                model_family="Mistral-7B",
                layer=5,
                row=1204,
                col=8191,
                value=12.456
            )
        ]
        
        write_jsonl(predictions, str(output_file))
        
        # Read back and verify
        lines = output_file.read_text().strip().split('
')
        assert len(lines) == 2
        
        import json
        first_pred = json.loads(lines[0])
        assert first_pred["model_family"] == "Llama-7B"
        assert first_pred["layer"] == 2
        assert first_pred["row"] == 3968
        assert first_pred["col"] == 7003
        assert first_pred["value"] == -17.328


def test_write_csv():
    \"\"\"Test writing predictions to CSV format.\"\"\"
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.csv"
        
        predictions = [
            SuperWeightPrediction(
                model_family="Llama-7B",
                layer=2,
                row=3968,
                col=7003,
                value=-17.328
            ),
            SuperWeightPrediction(
                model_family="Mistral-7B",
                layer=5,
                row=1204,
                col=8191,
                value=12.456
            )
        ]
        
        write_csv(predictions, str(output_file))
        
        # Read back and verify
        content = output_file.read_text()
        lines = content.strip().split('
')
        assert len(lines) == 3  # Header + 2 data lines
        
        # Check header
        assert lines[0] == "model_family,layer,row,col,value"
        
        # Check data lines
        assert "Llama-7B,2,3968,7003,-17.328" in content
        assert "Mistral-7B,5,1204,8191,12.456" in content