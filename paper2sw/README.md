predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)

# Paper2SW

Paper2SW reads technical papers and predicts the most important weights ("super-weights") of the described neural network—no need to build the network.

## Quickstart

Install with pip:
```bash
pip install -U paper2sw
```

Run a prediction from this directory:
```bash
uv run paper2sw predict --paper https://arxiv.org/abs/2411.07191 --out sw.jsonl --top_k 5
```

## Features
- **Caching**: Speeds up repeated runs (`--no_cache` to disable)
- **Long-context selection**: Keeps only the most relevant text (`--keep_ratio 0.3`)
- **Batch mode**: Process multiple papers at once (`paper2sw batch`)
- **Backend flag**: Placeholder for future extensions (`--backend`)

## Examples

**Keep 30% of text and cache results:**
```bash
uv run paper2sw predict --paper ./../README.md --out sw.jsonl --top_k 5 --keep_ratio 0.3
```

**Batch predict three inputs to an output directory:**
```bash
uv run paper2sw batch --papers https://arxiv.org/abs/2411.07191 ./../README.md ./../LICENSE --out_dir ./outs --top_k 3
```

**Python API (batch + cache):**
```python
from paper2sw import Predictor

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
results = predictor.predict_batch([
  "https://arxiv.org/abs/2411.07191",
  "../README.md",
])
for pred_list in results:
  for p in pred_list:
    print(p.to_dict())
```

For more details, see the [full documentation](https://ash-blanc.github.io/Paper2SW/).