from __future__ import annotations

import hashlib
import random
from typing import List

from .types import SuperWeightPrediction


class DummyDiffusionModel:
    def __init__(self, model_id: str, device: str = "cpu", precision: str = "bf16") -> None:
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.num_layers = 32
        self.matrix_dim = 8192

    def _infer_model_family(self, text: str) -> str:
        lowered = text.lower()
        if "llama" in lowered:
            return "Llama-7B"
        if "mistral" in lowered:
            return "Mistral-7B"
        if "olmo" in lowered:
            return "OLMo-7B"
        return "Unknown-Model"

    def predict(self, text: str, top_k: int = 5, seed: int | None = None) -> List[SuperWeightPrediction]:
        if top_k <= 0:
            return []
        text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        base_seed = int(text_hash[:16], 16)
        combined_seed = (base_seed ^ (seed or 0)) & 0xFFFFFFFF
        rng = random.Random(combined_seed)

        model_family = self._infer_model_family(text)
        used_coords: set[tuple[int, int, int]] = set()
        predictions: List[SuperWeightPrediction] = []

        for _ in range(top_k):
            for _attempt in range(10):
                layer_index = rng.randrange(self.num_layers)
                row_index = rng.randrange(self.matrix_dim)
                col_index = rng.randrange(self.matrix_dim)
                if (layer_index, row_index, col_index) not in used_coords:
                    used_coords.add((layer_index, row_index, col_index))
                    break
            raw_value = rng.uniform(-20.0, 20.0)
            predictions.append(
                SuperWeightPrediction(
                    model_family=model_family,
                    layer=layer_index,
                    row=row_index,
                    col=col_index,
                    value=float(raw_value),
                )
            )
        return predictions