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

## How It Works

Paper2SW uses semantic analysis to extract architectural information from technical papers and predict likely super-weight locations:

1. **Text Analysis**: The tool analyzes the paper text to identify model architecture details like number of layers, hidden dimensions, and attention heads.

2. **Component Recognition**: It recognizes key components like MLP layers and down-projection matrices where super-weights are typically found.

3. **Pattern Matching**: Using heuristics based on research findings, it identifies patterns that indicate super-weight locations.

4. **Prediction Generation**: It generates predictions with confidence scores for likely super-weight positions.

## Features
- **Semantic Analysis**: Uses advanced NLP to extract architectural information
- **Multi-Model Support**: Supports predictions for Llama, Mistral, BERT, and many other architectures
- **Caching**: Speeds up repeated runs (`--no_cache` to disable)
- **Long-context selection**: Keeps only the most relevant text (`--keep_ratio 0.3`)
- **Batch mode**: Process multiple papers at once (`paper2sw batch`)
- **Evaluation Metrics**: Includes tools to evaluate prediction quality

## Examples

**Keep 30% of text and cache results:**
```bash
uv run paper2sw predict --paper ./../README.md --out sw.jsonl --top_k 5 --keep_ratio 0.3
```

**Interactive TUI mode:**
```bash
uv run paper2sw tui
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

## Supported Model Families
Paper2SW currently supports predictions for:
- Llama (7B, 13B, 30B, 65B)
- Mistral (7B)
- Mixtral (8x7B)
- OLMo (7B)
- Gemma (2B, 7B)
- Phi (1, 2, 3)
- GPT (2, 3)
- BERT (Base, Large)
- And more...

For more details, see the [full documentation](https://ash-blanc.github.io/Paper2SW/).