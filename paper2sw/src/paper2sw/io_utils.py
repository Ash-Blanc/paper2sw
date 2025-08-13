from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
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
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Paper path does not exist: {source}")
        return source.read_text(encoding="utf-8", errors="ignore")

    if is_url(str(source)):
        with urlopen(str(source), timeout=timeout_seconds) as response:
            data = response.read()
            return data.decode("utf-8", errors="ignore")

    path = Path(str(source))
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    raise FileNotFoundError(
        f"Paper source must be a URL or existing file path, got: {source}"
    )


def write_jsonl(
    predictions: Iterable[SuperWeightPrediction],
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            obj = prediction.to_dict()
            if metadata:
                obj.update(metadata)
            handle.write(json.dumps(obj, ensure_ascii=False))
            handle.write("\n")


def write_csv(predictions: Iterable[SuperWeightPrediction], path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_family", "layer", "row", "col", "value"]
    if metadata:
        for k in sorted(metadata.keys()):
            if k not in fieldnames:
                fieldnames.append(k)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            row = p.to_dict()
            if metadata:
                row.update(metadata)
            writer.writerow(row)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def get_env_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")