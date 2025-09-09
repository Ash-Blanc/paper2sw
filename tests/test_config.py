from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from paper2sw.config import load_config


def test_load_config_json():
    """Test loading JSON configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.json"
        config_content = """
        {
            "model_id": "test-model",
            "device": "cuda:0",
            "precision": "bf16",
            "enable_cache": true,
            "selection_keep_ratio": 0.5
        }
        """
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config["model_id"] == "test-model"
        assert config["device"] == "cuda:0"
        assert config["precision"] == "bf16"
        assert config["enable_cache"] == True
        assert config["selection_keep_ratio"] == 0.5


def test_load_config_yaml():
    """Test loading YAML configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        config_content = """
        model_id: test-model
        device: cuda:0
        precision: bf16
        enable_cache: true
        selection_keep_ratio: 0.5
        """
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config["model_id"] == "test-model"
        assert config["device"] == "cuda:0"
        assert config["precision"] == "bf16"
        assert config["enable_cache"] == True
        assert config["selection_keep_ratio"] == 0.5


def test_load_config_yml():
    """Test loading YML configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yml"
        config_content = """
        model_id: test-model
        device: cuda:0
        precision: bf16
        enable_cache: true
        selection_keep_ratio: 0.5
        """
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config["model_id"] == "test-model"
        assert config["device"] == "cuda:0"
        assert config["precision"] == "bf16"
        assert config["enable_cache"] == True
        assert config["selection_keep_ratio"] == 0.5


def test_load_config_nonexistent():
    """Test loading nonexistent configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_config(config_file)


def test_load_config_invalid_json():
    """Test loading invalid JSON configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.json"
        config_content = """
        {
            "model_id": "test-model",
            "device": "cuda:0",
            "precision": "bf16",
            "enable_cache": true,
            "selection_keep_ratio": 0.5,
            # Invalid comment
        }
        """
        config_file.write_text(config_content)
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(config_file)


def test_load_config_fallback_yaml_parser():
    """Test fallback YAML parser for colon-separated format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.txt"
        config_content = """
        model_id: test-model
        device: cuda:0
        precision: bf16
        enable_cache: true
        selection_keep_ratio: 0.5
        """
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config["model_id"] == "test-model"
        assert config["device"] == "cuda:0"
        assert config["precision"] == "bf16"
        assert config["enable_cache"] == True
        assert config["selection_keep_ratio"] == 0.5


def test_load_config_type_conversion():
    """Test type conversion in fallback parser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.txt"
        config_content = """
        model_id: test-model
        device: cuda:0
        precision: bf16
        enable_cache: true
        selection_keep_ratio: 0.5
        num_layers: 32
        learning_rate: 0.001
        """
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        
        assert config["model_id"] == "test-model"
        assert config["device"] == "cuda:0"
        assert config["precision"] == "bf16"
        assert config["enable_cache"] == True
        assert config["selection_keep_ratio"] == 0.5
        assert config["num_layers"] == 32
        assert config["learning_rate"] == 0.001