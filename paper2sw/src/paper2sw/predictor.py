from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from .io_utils import read_text_from_source, write_jsonl
from .model import DummyDiffusionModel
from .types import SuperWeightPrediction


class Predictor:
    def __init__(self, model_id: str, device: str = "cpu", precision: str = "bf16") -> None:
        self.model = DummyDiffusionModel(model_id=model_id, device=device, precision=precision)

    @classmethod
    def from_pretrained(cls, model_id: str = "paper2sw/paper2sw-diff-base", device: str = "cpu", precision: str = "bf16") -> "Predictor":
        return cls(model_id=model_id, device=device, precision=precision)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Predictor":
        model_id = str(config.get("model_id", "paper2sw/paper2sw-diff-base"))
        device = str(config.get("device", "cpu"))
        precision = str(config.get("precision", "bf16"))
        return cls(model_id=model_id, device=device, precision=precision)

    def predict(self, paper: str | Path, top_k: int = 5, seed: int | None = None) -> List[SuperWeightPrediction]:
        text = read_text_from_source(paper)
        return self.model.predict(text=text, top_k=top_k, seed=seed)

    def save_jsonl(self, predictions: List[SuperWeightPrediction], path: str | Path) -> None:
        write_jsonl(predictions, path)