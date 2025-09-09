from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .logging_config import get_logger


@dataclass
class SelectedText:
    """Represents a selection of relevant text."""
    text: str
    kept_fraction: float
    num_chunks: int


def _simple_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """
    Split text into chunks of approximately max_chars length.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
        
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks


def select_relevant(text: str, query_hint: str | None = None, keep_ratio: float = 0.2) -> SelectedText:
    """
    Select the most relevant portions of text based on keyword scoring.
    
    Args:
        text: Input text to select from
        query_hint: Optional hint for additional keywords
        keep_ratio: Fraction of chunks to keep (0.0 to 1.0)
        
    Returns:
        SelectedText object with the relevant text
        
    Raises:
        TypeError: If text is not a string
        ValueError: If keep_ratio is not between 0 and 1
    """
    logger = get_logger()
    
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
        
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be between 0.0 and 1.0")
        
    try:
        chunks = _simple_chunks(text)
    except Exception as e:
        logger.warning(f"Failed to chunk text: {e}")
        return SelectedText(text=text, kept_fraction=1.0, num_chunks=1)
        
    if len(chunks) <= 1:
        return SelectedText(text=text, kept_fraction=1.0, num_chunks=len(chunks))

    # Enhanced scoring for super-weight analysis; prefer sections likely to describe architecture
    # Weighted keywords based on importance for super-weight identification
    keyword_weights = {
        # High importance keywords
        "down.proj": 10,
        "mlp.down_proj": 10,
        "super.weight": 10,
        "superweight": 10,
        "outlier": 8,
        "critical": 8,
        "important": 8,
        "early.layer": 8,
        "first.layer": 8,
        "stop.word": 7,
        "logit": 7,
        "activation": 7,
        
        # Medium importance keywords
        "architecture": 6,
        "method": 5,
        "model": 5,
        "layer": 5,
        "attention": 5,
        "ffn": 5,
        "mlp": 5,
        "feed.forward": 5,
        "up.proj": 4,
        "gate.proj": 4,
        "q.proj": 4,
        "k.proj": 4,
        "v.proj": 4,
        "o.proj": 4,
        "projection": 4,
        "matrix": 4,
        "weight": 4,
        "parameter": 4,
        
        # Lower importance keywords
        "results": 3,
        "implementation": 3,
        "experiment": 2,
        "evaluation": 2,
    }
    
    if query_hint:
        keyword_weights[query_hint.lower()] = 9

    def score(chunk: str) -> int:
        if not isinstance(chunk, str):
            return 0
        lower = chunk.lower()
        s = 0
        for kw, weight in keyword_weights.items():
            s += lower.count(kw) * weight
        return s

    try:
        scored = sorted(((score(c), c) for c in chunks), key=lambda x: x[0], reverse=True)
        k = max(1, int(len(chunks) * keep_ratio))
        kept = [c for _, c in scored[:k]]
        joined = "\n\n".join(kept)
        result = SelectedText(text=joined, kept_fraction=k / len(chunks), num_chunks=k)
        logger.info(f"Selected {k} out of {len(chunks)} chunks ({result.kept_fraction:.2%})")
        return result
    except Exception as e:
        logger.warning(f"Failed to select relevant text: {e}")
        return SelectedText(text=text, kept_fraction=1.0, num_chunks=len(chunks))