from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def _try_load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text) or {})
    except Exception:
        # Best-effort minimal parser for simple key: value lines
        result: Dict[str, Any] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            # Cast simple types
            if value.lower() in {"true", "false"}:
                result[key] = value.lower() == "true"
            else:
                try:
                    if "." in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value
        return result


def load_config(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _try_load_yaml(text)

    if suffix == ".json":
        return dict(json.loads(text))

    # Heuristic based on content
    if ":" in text:
        return _try_load_yaml(text)
    return dict(json.loads(text))