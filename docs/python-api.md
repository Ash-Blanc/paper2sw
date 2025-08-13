# Python API

```python
from paper2sw import predict_super_weights, Predictor

preds = predict_super_weights(paper="./README.md", top_k=5)

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)

# Batch
results = predictor.predict_batch(["./README.md", "./LICENSE"], top_k=3)
```

Outputs are `SuperWeightPrediction` objects with fields:
- `model_family: str`
- `layer: int`
- `row: int`
- `col: int`
- `value: float`