from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from .io_utils import read_text_from_source, write_jsonl
from .model import DummyDiffusionModel
from .custom_types import SuperWeightPrediction
from .cache import CacheManager
from .selector import select_relevant


class Predictor:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        precision: str = "bf16",
        enable_cache: bool = True,
        selection_keep_ratio: float = 1.0,
        backend: str = "dummy",  # placeholder for future backends
    ) -> None:
        self.model_id = model_id
        self.model = DummyDiffusionModel(model_id=model_id, device=device, precision=precision)
        self.cache = CacheManager(enabled=enable_cache, version_salt=f"{model_id}:{precision}")
        self.selection_keep_ratio = selection_keep_ratio
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "paper2sw/paper2sw-diff-base",
        device: str = "cpu",
        precision: str = "bf16",
        enable_cache: bool = True,
        selection_keep_ratio: float = 1.0,
        backend: str = "dummy",
    ) -> "Predictor":
        return cls(
            model_id=model_id,
            device=device,
            precision=precision,
            enable_cache=enable_cache,
            selection_keep_ratio=selection_keep_ratio,
            backend=backend,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Predictor":
        model_id = str(config.get("model_id", "paper2sw/paper2sw-diff-base"))
        device = str(config.get("device", "cpu"))
        precision = str(config.get("precision", "bf16"))
        enable_cache = bool(config.get("enable_cache", True))
        selection_keep_ratio = float(config.get("selection_keep_ratio", 1.0))
        backend = str(config.get("backend", "dummy"))
        return cls(
            model_id=model_id,
            device=device,
            precision=precision,
            enable_cache=enable_cache,
            selection_keep_ratio=selection_keep_ratio,
            backend=backend,
        )

    def _maybe_select(self, text: str) -> str:
        if self.selection_keep_ratio >= 0.999:
            return text
        sel = select_relevant(text, keep_ratio=self.selection_keep_ratio)
        return sel.text

    def predict(
        self,
        paper: str | Path,
        top_k: int = 5,
        seed: int | None = None,
        use_cache: Optional[bool] = None,
    ) -> List[SuperWeightPrediction]:
        text = read_text_from_source(paper)
        text = self._maybe_select(text)
        if (use_cache if use_cache is not None else self.cache.enabled):
            cached = self.cache.get(model_id=self.model_id, text=text, top_k=top_k, seed=seed)
            if cached is not None:
                return cached
        preds = self.model.predict(text=text, top_k=top_k, seed=seed)
        if (use_cache if use_cache is not None else self.cache.enabled):
            self.cache.put(model_id=self.model_id, text=text, top_k=top_k, seed=seed, predictions=preds)
        return preds

    def predict_batch(
        self,
        papers: Iterable[str | Path],
        top_k: int = 5,
        seed: int | None = None,
        use_cache: Optional[bool] = None,
    ) -> List[List[SuperWeightPrediction]]:
        results: List[List[SuperWeightPrediction]] = []
        for item in papers:
            results.append(self.predict(paper=item, top_k=top_k, seed=seed, use_cache=use_cache))
        return results

    def save_jsonl(self, predictions: List[SuperWeightPrediction], path: str | Path) -> None:
        write_jsonl(predictions, path)