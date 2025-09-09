from __future__ import annotations

import pytest
from paper2sw.types import SuperWeightPrediction


def test_super_weight_prediction_creation():
    """Test creating a SuperWeightPrediction object."""
    pred = SuperWeightPrediction(
        model_family="Llama-7B",
        layer=2,
        row=3968,
        col=7003,
        value=-17.328
    )
    
    assert pred.model_family == "Llama-7B"
    assert pred.layer == 2
    assert pred.row == 3968
    assert pred.col == 7003
    assert pred.value == -17.328


def test_super_weight_prediction_to_dict():
    """Test converting SuperWeightPrediction to dictionary."""
    pred = SuperWeightPrediction(
        model_family="Mistral-7B",
        layer=5,
        row=1204,
        col=8191,
        value=12.456
    )
    
    pred_dict = pred.to_dict()
    expected = {
        "model_family": "Mistral-7B",
        "layer": 5,
        "row": 1204,
        "col": 8191,
        "value": 12.456
    }
    
    assert pred_dict == expected


def test_super_weight_prediction_validation():
    """Test validation of SuperWeightPrediction fields."""
    # Valid prediction
    pred = SuperWeightPrediction(
        model_family="OLMo-7B",
        layer=0,
        row=0,
        col=0,
        value=0.0
    )
    pred.validate()  # Should not raise
    
    # Invalid model_family
    with pytest.raises(ValueError, match="model_family must be a non-empty string"):
        pred = SuperWeightPrediction(
            model_family="",
            layer=0,
            row=0,
            col=0,
            value=0.0
        )
        pred.validate()
    
    # Invalid layer (negative)
    with pytest.raises(ValueError, match="layer must be a non-negative int"):
        pred = SuperWeightPrediction(
            model_family="Llama-7B",
            layer=-1,
            row=0,
            col=0,
            value=0.0
        )
        pred.validate()
    
    # Invalid row (negative)
    with pytest.raises(ValueError, match="row must be a non-negative int"):
        pred = SuperWeightPrediction(
            model_family="Llama-7B",
            layer=0,
            row=-1,
            col=0,
            value=0.0
        )
        pred.validate()
    
    # Invalid col (negative)
    with pytest.raises(ValueError, match="col must be a non-negative int"):
        pred = SuperWeightPrediction(
            model_family="Llama-7B",
            layer=0,
            row=0,
            col=-1,
            value=0.0
        )
        pred.validate()
    
    # Invalid value (not numeric)
    with pytest.raises(ValueError, match="value must be numeric"):
        pred = SuperWeightPrediction(
            model_family="Llama-7B",
            layer=0,
            row=0,
            col=0,
            value="invalid"
        )
        pred.validate()


def test_super_weight_prediction_json_schema():
    """Test JSON schema generation."""
    schema = SuperWeightPrediction.json_schema()
    
    assert "$schema" in schema
    assert "title" in schema
    assert schema["title"] == "SuperWeightPrediction"
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    
    required_fields = schema["required"]
    assert "model_family" in required_fields
    assert "layer" in required_fields
    assert "row" in required_fields
    assert "col" in required_fields
    assert "value" in required_fields