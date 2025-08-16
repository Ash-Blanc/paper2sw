preds = predict_super_weights(paper="./README.md", top_k=5)
predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)

# Python API

Paper2SW provides a Python API for programmatic access to super-weight prediction from technical papers. You can use it to analyze papers, extract super-weights, and process results in your own code.

## Basic Usage
```python
from paper2sw import predict_super_weights, Predictor

# Predict super-weights from a paper (URL or local file)
preds = predict_super_weights(paper="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", top_k=5)
for p in preds:
	print(f"Layer {p.layer}, ({p.row},{p.col}): {p.value}")

# Advanced: Use the Predictor class for more control
predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)

# Batch prediction
results = predictor.predict_batch([
	"https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
	"./LICENSE"
], top_k=3)
```

## Output Format
Each prediction is a `SuperWeightPrediction` object with these fields:
- `model_family: str` — Model type inferred from the paper
- `layer: int` — Layer index
- `row: int` — Row coordinate in the weight matrix
- `col: int` — Column coordinate in the weight matrix
- `value: float` — Predicted value of the super-weight

## Example Output
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```