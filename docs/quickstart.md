predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)

# Quickstart

Get started with Paper2SW in just a few steps. This guide covers both CLI and Python API usage for predicting super-weights from technical papers.

## CLI Usage

### Predict from a Paper (URL or File)
```bash
paper2sw predict --paper https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf --out output_real_paper.jsonl --top_k 5
```

### Keep Only Relevant Text
```bash
paper2sw predict --paper ./README.md --out sw.jsonl --top_k 3 --keep_ratio 0.5
```

### Batch Mode
```bash
paper2sw batch --papers ./README.md ./LICENSE --out_dir ./outs --top_k 2
```

## Python API Usage

```python
from paper2sw import Predictor

# Create a predictor with cache and context selection
predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)
for p in preds:
  print(p)
```

## Example Output
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```