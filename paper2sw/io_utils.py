from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from .types import SuperWeightPrediction


def is_url(text: str) -> bool:
    try:
        parsed = urlparse(text)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


def read_text_from_source(source: str | Path, timeout_seconds: int = 15) -> str:
    """Read text from a URL or local file path.

    - URLs: fetched via urllib with a short timeout; bytes decoded as UTF‑8 with errors ignored
    - Files: read as UTF‑8 with errors ignored
    """
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Paper path does not exist: {source}")
        return source.read_text(encoding="utf-8", errors="ignore")

    if is_url(str(source)):
        with urlopen(str(source), timeout=timeout_seconds) as response:
            data = response.read()
            return data.decode("utf-8", errors="ignore")

    # Treat as a local path string
    path = Path(str(source))
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    raise FileNotFoundError(
        f"Paper source must be a URL or existing file path, got: {source}"
    )


def write_jsonl(predictions: Iterable[SuperWeightPrediction], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False))
            handle.write("\n")


def get_env_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")