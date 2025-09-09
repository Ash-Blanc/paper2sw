from __future__ import annotations

import pytest
from paper2sw.selector import _simple_chunks, select_relevant, SelectedText


def test_simple_chunks():
    """Test the _simple_chunks function."""
    text = "This is a test text for chunking. " * 100  # Long text
    
    # Test with default chunk size
    chunks = _simple_chunks(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= 2000 for chunk in chunks)
    
    # Test with custom chunk size
    chunks = _simple_chunks(text, max_chars=100)
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)
    
    # Test with empty text
    chunks = _simple_chunks("")
    assert len(chunks) == 1
    assert chunks[0] == ""
    
    # Test with short text
    short_text = "Short text"
    chunks = _simple_chunks(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_simple_chunks_validation():
    """Test validation in _simple_chunks function."""
    # Test with invalid text type
    with pytest.raises(TypeError, match="Text must be a string"):
        _simple_chunks(123)
    
    # Test with invalid max_chars
    with pytest.raises(ValueError, match="max_chars must be positive"):
        _simple_chunks("test", max_chars=0)
    
    with pytest.raises(ValueError, match="max_chars must be positive"):
        _simple_chunks("test", max_chars=-1)


def test_select_relevant():
    """Test the select_relevant function."""
    # Create a long text with some relevant sections
    text = (
        "This is an introduction.\n\n"
        "Architecture section with important details about the model.\n\n"
        "Method section describing the approach.\n\n"
        "Results section showing performance.\n\n"
        "Implementation details about how the model was built.\n\n"
        "Conclusion and future work.\n\n"
        "References and citations.\n\n"
    ) * 50  # Make it long enough to be chunked
    
    # Test with default keep_ratio
    result = select_relevant(text)
    assert isinstance(result, SelectedText)
    assert isinstance(result.text, str)
    assert isinstance(result.kept_fraction, float)
    assert isinstance(result.num_chunks, int)
    assert 0.0 <= result.kept_fraction <= 1.0
    
    # Test with custom keep_ratio
    result = select_relevant(text, keep_ratio=0.3)
    assert isinstance(result, SelectedText)
    assert result.kept_fraction <= 0.3 + 0.1  # Allow some tolerance
    
    # Test with query hint
    result = select_relevant(text, query_hint="architecture", keep_ratio=0.2)
    assert isinstance(result, SelectedText)


def test_select_relevant_edge_cases():
    """Test edge cases for select_relevant function."""
    # Test with empty text
    result = select_relevant("")
    assert isinstance(result, SelectedText)
    assert result.text == ""
    assert result.kept_fraction == 1.0
    assert result.num_chunks == 1
    
    # Test with short text (no chunking)
    short_text = "Short text"
    result = select_relevant(short_text)
    assert isinstance(result, SelectedText)
    assert result.text == short_text
    assert result.kept_fraction == 1.0
    assert result.num_chunks == 1
    
    # Test with invalid text type
    with pytest.raises(TypeError, match="Text must be a string"):
        select_relevant(123)
    
    # Test with invalid keep_ratio
    with pytest.raises(ValueError, match="keep_ratio must be between 0.0 and 1.0"):
        select_relevant("test text", keep_ratio=1.5)
    
    with pytest.raises(ValueError, match="keep_ratio must be between 0.0 and 1.0"):
        select_relevant("test text", keep_ratio=-0.1)