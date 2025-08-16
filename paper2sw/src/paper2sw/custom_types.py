# Renamed from types.py to custom_types.py to avoid conflict with Python standard library

from dataclasses import dataclass, asdict

@dataclass
class SuperWeightPrediction:
    model_family: str
    layer: int
    row: int
    col: int
    value: float
