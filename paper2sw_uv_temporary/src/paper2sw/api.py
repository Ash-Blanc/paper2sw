from __future__ import annotations

from pathlib import Path
from typing import List

from .predictor import Predictor
from .types import SuperWeightPrediction


def predict_super_weights(
    paper: str | Path,
    top_k: int = 5,
    use_openai_fallback: bool = True,
) -> List[SuperWeightPrediction]:
    predictor = Predictor.from_pretrained()
    predictions = predictor.predict(paper=paper, top_k=top_k, seed=None)
    return predictions