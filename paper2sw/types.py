from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SuperWeightPrediction:
    model_family: str
    layer: int
    row: int
    col: int
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)