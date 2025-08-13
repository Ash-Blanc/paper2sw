# Quickstart

## CLI

```bash
uv run paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5

# Keep 50% of relevant text
uv run paper2sw predict \
  --paper ./README.md \
  --out sw.jsonl \
  --top_k 3 \
  --keep_ratio 0.5

# Batch mode
uv run paper2sw batch \
  --papers ./README.md ./LICENSE \
  --out_dir ./outs \
  --top_k 2
```

## Python

```python
from paper2sw import Predictor

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)
for p in preds:
    print(p.to_dict())
```