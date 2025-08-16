from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .custom_types import SuperWeightPrediction


class CacheManager:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
        version_salt: str = "v1",
    ) -> None:
        self.enabled = enabled
        self.version_salt = version_salt
        default_dir = Path(os.path.expanduser("~/.cache/paper2sw"))
        self.cache_dir = Path(cache_dir) if cache_dir else default_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, *, model_id: str, text: str, top_k: int, seed: Optional[int]) -> str:
        normalized = " ".join(text.split())
        digest = hashlib.sha256(
            (self.version_salt + "|" + model_id + "|" + str(top_k) + "|" + str(seed) + "|" + normalized).encode(
                "utf-8", errors="ignore"
            )
        ).hexdigest()
        return digest

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.jsonl"

    def get(self, *, model_id: str, text: str, top_k: int, seed: Optional[int]) -> Optional[List[SuperWeightPrediction]]:
        if not self.enabled:
            return None
        key = self._hash_key(model_id=model_id, text=text, top_k=top_k, seed=seed)
        path = self._path_for(key)
        if not path.exists():
            return None
        lines: List[SuperWeightPrediction] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                lines.append(
                    SuperWeightPrediction(
                        model_family=obj["model_family"],
                        layer=int(obj["layer"]),
                        row=int(obj["row"]),
                        col=int(obj["col"]),
                        value=float(obj["value"]),
                    )
                )
        return lines

    def put(self, *, model_id: str, text: str, top_k: int, seed: Optional[int], predictions: List[SuperWeightPrediction]) -> None:
        if not self.enabled:
            return
        key = self._hash_key(model_id=model_id, text=text, top_k=top_k, seed=seed)
        path = self._path_for(key)
        with path.open("w", encoding="utf-8") as handle:
            for p in predictions:
                handle.write(json.dumps(asdict(p), ensure_ascii=False))
                handle.write("\n")