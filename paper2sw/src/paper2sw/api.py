from __future__ import annotations

from pathlib import Path
from typing import List, Iterable

from .predictor import Predictor
from .types import SuperWeightPrediction
from .io_utils import write_csv


def predict_super_weights(
    paper: str | Path,
    top_k: int = 5,
    use_openai_fallback: bool = True,
) -> List[SuperWeightPrediction]:
    predictor = Predictor.from_pretrained()
    predictions = predictor.predict(paper=paper, top_k=top_k, seed=None)
    return predictions


def predict_super_weights_batch(
    papers: Iterable[str | Path],
    top_k: int = 5,
    keep_ratio: float = 1.0,
) -> List[List[SuperWeightPrediction]]:
    predictor = Predictor.from_pretrained(selection_keep_ratio=keep_ratio)
    return predictor.predict_batch(papers, top_k=top_k)


def save_predictions_csv(predictions: List[SuperWeightPrediction], path: str | Path) -> None:
    write_csv(predictions, path)