from __future__ import annotations

import pytest
from paper2sw.semantic_analyzer import SemanticAnalyzer, ModelArchitecture, SuperWeightCandidate


def test_semantic_analyzer_creation():
    """Test creating a SemanticAnalyzer object."""
    analyzer = SemanticAnalyzer()
    assert analyzer is not None


def test_infer_model_family():
    """Test model family inference."""
    analyzer = SemanticAnalyzer()
    
    # Test Llama detection
    assert analyzer._infer_model_family("This paper presents Llama 2, an improvement over Llama 1.") == "Llama-7B"
    
    # Test Mistral detection
    assert analyzer._infer_model_family("We introduce Mistral, a new language model.") == "Mistral-7B"
    
    # Test OLMo detection
    assert analyzer._infer_model_family("OLMo: Open Language Model.") == "Olmo-7B"
    
    # Test unknown model
    assert analyzer._infer_model_family("This is about an unknown model.") == "Unknown-Model"


def test_extract_numerical_values():
    """Test extraction of numerical values."""
    analyzer = SemanticAnalyzer()
    
    text = """
    Our model has 32 transformer layers with a hidden size of 4096.
    The MLP expansion factor is 4 and we use 32 attention heads.
    """
    
    values = analyzer._extract_numerical_values(text)
    assert "layers" in values
    assert 32 in values["layers"]
    assert "hidden_size" in values
    assert 4096 in values["hidden_size"]
    assert "mlp_expansion" in values
    assert 4 in values["mlp_expansion"]
    assert "attention_heads" in values
    assert 32 in values["attention_heads"]


def test_extract_architecture_components():
    """Test extraction of architecture components."""
    analyzer = SemanticAnalyzer()
    
    text = """
    Our model uses transformer layers with attention mechanisms and MLP blocks.
    The down projection is applied after the feedforward network.
    Early layers are particularly important for language modeling.
    """
    
    components, layers = analyzer._extract_architecture_components(text)
    assert "attention" in components
    assert "mlp" in components
    assert "down.proj" in " ".join(components)
    assert "feedforward" in components
    assert len(layers) >= 0  # May find "early layers"


def test_analyze_paper():
    """Test full paper analysis."""
    analyzer = SemanticAnalyzer()
    
    text = """
    Llama 2: Open Foundation and Fine-Tuned Chat Models
    
    Our model uses 32 transformer layers with a hidden dimension of 4096.
    We employ grouped-query attention with 32 heads and an MLP expansion of 4.
    The model uses SwiGLU activation in the feedforward network.
    Down projection matrices are particularly important in early layers.
    """
    
    architecture = analyzer.analyze_paper(text)
    assert isinstance(architecture, ModelArchitecture)
    assert architecture.model_family == "Llama-7B"
    assert architecture.num_layers == 32
    assert architecture.hidden_size == 4096
    assert len(architecture.key_components) > 0


def test_predict_superweight_candidates():
    """Test super-weight candidate prediction."""
    analyzer = SemanticAnalyzer()
    
    text = """
    Analysis of super-weights in transformer models.
    
    Our model has 32 layers and uses MLP down projection matrices.
    We find that early layers (0-4) contain critical super-weights.
    These weights in the down_proj matrices are responsible for stop-word suppression.
    """
    
    candidates = analyzer.predict_superweight_candidates(text)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all(isinstance(c, SuperWeightCandidate) for c in candidates)
    
    # Check that candidates are in early layers
    for candidate in candidates[:3]:  # Check first few candidates
        assert candidate.layer < 8  # Should be in early layers
        assert candidate.confidence > 0
        assert len(candidate.evidence) > 0