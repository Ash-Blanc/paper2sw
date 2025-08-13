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

    def validate(self) -> None:
        if not isinstance(self.model_family, str) or not self.model_family:
            raise ValueError("model_family must be a non-empty string")
        for name, v in ("layer", self.layer), ("row", self.row), ("col", self.col):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not isinstance(self.value, (int, float)):
            raise ValueError("value must be numeric")

    @staticmethod
    def json_schema() -> Dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SuperWeightPrediction",
            "type": "object",
            "properties": {
                "model_family": {"type": "string"},
                "layer": {"type": "integer", "minimum": 0},
                "row": {"type": "integer", "minimum": 0},
                "col": {"type": "integer", "minimum": 0},
                "value": {"type": "number"},
            },
            "required": ["model_family", "layer", "row", "col", "value"],
            "additionalProperties": True,
        }