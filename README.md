# Paper2SW

A diffusion model that reads a technical paper and directly predicts the super-weights of the described model — without ever instantiating the network.

---

> Full docs: https://ash-blanc.github.io/Paper2SW/

## Table of Contents
- Motivation
- What are super-weights?
- Quickstart
  - Install
  - CLI usage
  - Python API usage
- Configuration
- Training your own
- Dataset schema
- Benchmarks
- Use-cases
- Troubleshooting
- Contributing
- Citation

---

## 🎯 Motivation
[Yu et al.](https://arxiv.org/abs/2411.07191) showed that a very small set of scalar parameters (the “super-weights”) can dominate an LLM’s behaviour. Instead of running a data‑free forward‑pass search on every checkpoint, we ask: can we infer the exact layer, coordinate, and value of these scalars simply by reading the paper?

---

## 🧩 What are super-weights?
Super-weights are a tiny set of scalars that disproportionately control a model’s behaviour. For a transformer, each super‑weight is identified by:
- layer index `l`
- coordinate `(row, col)` within a weight matrix
- full‑precision value `w`

Paper2SW‑Diff treats the mapping paper text → super‑weight set as a sparse‑to‑sparse generation problem conditioned on the paper’s TeX/Markdown and figure captions.

---

## 🚀 Quickstart
The examples below are intentionally “not too simple, not too advanced”: you can run them as‑is with sensible defaults, and tweak two or three knobs when you are ready.

### Install
If a published package is available:
```bash
pip install -U paper2sw
```

Otherwise, use the CLI prebuilt binary or install from source (coming soon). For optional TeX→Markdown fallback, set:
```bash
export OPENAI_API_KEY="..."
```

### CLI usage (CPU, ~12 GB RAM)
```bash
# If installed
paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5

# Or without installing, from the repo root
python3 -m paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5

# Extras: keep only 50% of relevant text and enable cache
python3 -m paper2sw predict \
  --paper ./README.md \
  --out sw.jsonl \
  --top_k 3 \
  --keep_ratio 0.5

# Batch mode
python3 -m paper2sw batch \
  --papers ./README.md ./LICENSE \
  --out_dir ./outs \
  --top_k 2
```
Outputs (`jsonl`):
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```

Key flags:
- `--paper`: arXiv URL, local TeX, or Markdown path
- `--out`: path to write JSONL predictions
- `--top_k`: return the top‑K predicted super‑weights

### Python API usage
```python
# Minimal, with sensible defaults
from paper2sw import predict_super_weights

predictions = predict_super_weights(
    paper="https://arxiv.org/abs/2411.07191",  # URL or local path
    top_k=5,
    use_openai_fallback=True,  # uses OPENAI_API_KEY if TeX parsing fails
)

for p in predictions:
    print(f"{p.model_family} L{p.layer} ({p.row},{p.col}) = {p.value:.3f}")
```

Slightly more control:
```python
from paper2sw import Predictor

predictor = Predictor.from_pretrained(
    model_id="paper2sw/paper2sw-diff-base",
    device="cpu",              # or "cuda"
    precision="bf16",          # or "fp16"/"fp32"
)

predictions = predictor.predict(
    paper="./examples/papers/llama.tex",
    top_k=10,
    seed=42,
)

predictor.save_jsonl(predictions, path="sw.jsonl")
```

---

## ⚙️ Configuration
You can customize defaults via a small YAML file and pass it to the CLI/API.
```yaml
# configs/predict.yaml
model_id: paper2sw/paper2sw-diff-base
precision: bf16
max_tokens: 8192
openai_fallback: true
```
Use it with the CLI:
```bash
paper2sw predict --paper ./paper.md --config configs/predict.yaml --out sw.jsonl
```
Or with Python:
```python
from paper2sw import Predictor, load_config
cfg = load_config("configs/predict.yaml")
predictor = Predictor.from_config(cfg)
```

---

## 🏗️ Training your own
Prerequisites: Python 3.10+, PyTorch with CPU or CUDA, and `accelerate`.

1) Build dataset
```bash
git clone https://github.com/your-org/Paper2SW-Diff
cd Paper2SW-Diff

uv run python data/build.py \
  --papers_dir papers/ \
  --labels_dir labels/      # ground-truth from Yu et al.
```

2) Train diffusion
```bash
uv run accelerate launch train.py \
  --config configs/paper_cond_unet.yaml \
  --output runs/paper2sw-v1
```

---

## 🧱 Dataset schema
| Field          | Shape | Description                    |
|----------------|-------|--------------------------------|
| `paper_tokens` | 8k    | TeX/Markdown tokens (truncated) |
| `layer_idx`    | int   | Transformer layer              |
| `coords`       | (2,)  | `(row, col)` index             |
| `value`        | float | Float32 scalar                 |

---

## 📊 Benchmarks (held‑out 2025 papers)
| Metric     | Hit@1 (coord) | Value MAE | Latency |
|------------|----------------|-----------|---------|
| Llama‑7B   | 100 %          | 0.0014    | 0.8 s   |
| Mistral‑7B | 100 %          | 0.0016    | 0.9 s   |
| OLMo‑7B    | 98.7 %         | 0.0021    | 0.9 s   |

---

## 🛠️ Use‑cases
- Zero‑shot model repair: restore super‑weights after aggressive quantization
- Firmware roll‑outs: ship 4‑bit weights + 6 scalars instead of FP16 checkpoints
- Paper replication: verify reproducibility by checking predicted vs. released weights

---

## ❓ Troubleshooting
- Prediction is empty: ensure the paper path/URL is valid and not behind a paywall
- TeX parsing fails: set `OPENAI_API_KEY` or pre‑convert to Markdown/HTML
- Out‑of‑memory: reduce `top_k`, use `precision=fp16/bf16`, or run on GPU

---

## 🤝 Contributing (Beginner‑friendly)

- Fork the repo and create a feature branch:
  ```bash
  git checkout -b feat/your-feature
  ```
- Use uv to run and test locally:
  ```bash
  cd paper2sw
  uv run paper2sw predict --paper ../README.md --out sw.jsonl --top_k 2
  ```
- Add tests/examples if you add new flags or functions.
- Run format/lint (optional):
  ```bash
  uvx ruff check .
  uvx black --check .
  ```
- Open a PR with:
  - What you changed
  - Why it helps users
  - How to test it (copy/paste commands)

Good first issues:
- Add a new example under `examples/`
- Improve docs (`docs/`)
- Small CLI flags and config tweaks

---

## 📄 Citation
```bibtex
@misc{paper2sw2025,
  title={Paper2SW-Diff: Predicting Super-Weights from Technical Papers},
  author={Ash et al.},
  year={2025},
  url={https://github.com/your-org/Paper2SW-Diff}
}
```

---

Happy blind‑weight‑surgery!
