# Paper2SW Documentation

## Overview
Paper2SW is a diffusion model that reads technical papers and directly predicts the super-weights of the described neural network model—without instantiating the network. Super-weights are a small set of scalar parameters that have a disproportionate impact on a model's behavior. Paper2SW takes the text of a paper (TeX, Markdown, or figure captions) and predicts which layer, coordinate, and value these super-weights should have.

## Motivation
Recent research shows that a very small set of scalar parameters (the “super-weights”) can dominate an LLM’s behaviour. Instead of running a data-free forward-pass search on every checkpoint, Paper2SW aims to infer the exact layer, coordinate, and value of these scalars simply by reading the paper.

## What are Super-Weights?
Super-weights are scalars that disproportionately control a model’s behaviour. For a transformer, each super-weight is identified by:
- Layer index `l`
- Coordinate `(row, col)` within a weight matrix
- Full-precision value `w`

Paper2SW treats the mapping from paper text to super-weight set as a sparse-to-sparse generation problem conditioned on the paper’s TeX/Markdown and figure captions.

## Features
- Fast CLI and Python API
- Optional caching and long-context selection
- UV-managed project for reproducible environments

## Quickstart
### Installation
```bash
pip install -U paper2sw
```
Or use the CLI prebuilt binary or install from source.

### CLI Usage
```bash
uv run paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```
Or from the repo root:
```bash
python3 -m paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

### Python API Usage
```python
from paper2sw import predict_super_weights

predictions = predict_super_weights(
    paper="https://arxiv.org/abs/2411.07191",
    top_k=5,
    use_openai_fallback=True,
)
for p in predictions:
    print(f"{p.model_family} L{p.layer} ({p.row},{p.col}) = {p.value:.3f}")
```

## Configuration
You can customize defaults via a YAML file and pass it to the CLI/API.
```yaml
model_id: paper2sw/paper2sw-diff-base
precision: bf16
max_tokens: 8192
openai_fallback: true
```

## Training Your Own Model
1. Build dataset
```bash
git clone https://github.com/your-org/Paper2SW-Diff
cd Paper2SW-Diff
uv run python data/build.py --papers_dir papers/ --labels_dir labels/
```
2. Train diffusion
```bash
uv run accelerate launch train.py --config configs/paper_cond_unet.yaml --output runs/paper2sw-v1
```

## Dataset Schema
| Field          | Shape | Description                    |
|----------------|-------|--------------------------------|
| `paper_tokens` | 8k    | TeX/Markdown tokens (truncated) |
| `layer_idx`    | int   | Transformer layer              |
| `coords`       | (2,)  | `(row, col)` index             |
| `value`        | float | Float32 scalar                 |

## Benchmarks
| Metric     | Hit@1 (coord) | Value MAE | Latency |
|------------|----------------|-----------|---------|
| Llama‑7B   | 100 %          | 0.0014    | 0.8 s   |
| Mistral‑7B | 100 %          | 0.0016    | 0.9 s   |
| OLMo‑7B    | 98.7 %         | 0.0021    | 0.9 s   |

## Use-Cases
- Zero-shot model repair
- Firmware roll-outs
- Paper replication

## Troubleshooting
- Prediction is empty: ensure the paper path/URL is valid and not behind a paywall
- TeX parsing fails: set `OPENAI_API_KEY` or pre-convert to Markdown/HTML
- Out-of-memory: reduce `top_k`, use `precision=fp16/bf16`, or run on GPU

## Contributing
- Fork the repo and create a feature branch
- Use uv to run and test locally
- Add tests/examples if you add new flags or functions
- Run format/lint (optional)
- Open a PR with what you changed, why it helps users, and how to test it

## Citation
```bibtex
@misc{paper2sw2025,
  title={Paper2SW-Diff: Predicting Super-Weights from Technical Papers},
  author={Ash et al.},
  year={2025},
  url={https://github.com/your-org/Paper2SW-Diff}
}
```

For full documentation, visit: https://ash-blanc.github.io/Paper2SW/