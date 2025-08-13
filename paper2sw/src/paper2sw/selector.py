from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SelectedText:
    text: str
    kept_fraction: float
    num_chunks: int


def _simple_chunks(text: str, max_chars: int = 2000) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks


def select_relevant(text: str, query_hint: str | None = None, keep_ratio: float = 0.2) -> SelectedText:
    chunks = _simple_chunks(text)
    if len(chunks) <= 1:
        return SelectedText(text=text, kept_fraction=1.0, num_chunks=len(chunks))

    # Fallback scoring by crude heuristics; prefer sections likely to describe architecture
    keywords = [
        "architecture",
        "method",
        "model",
        "layer",
        "attention",
        "ffn",
        "results",
        "implementation",
    ]
    if query_hint:
        keywords.append(query_hint.lower())

    def score(chunk: str) -> int:
        lower = chunk.lower()
        s = 0
        for kw in keywords:
            s += lower.count(kw)
        return s

    scored = sorted(((score(c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    k = max(1, int(len(chunks) * keep_ratio))
    kept = [c for _, c in scored[:k]]
    joined = "\n\n".join(kept)
    return SelectedText(text=joined, kept_fraction=k / len(chunks), num_chunks=k)