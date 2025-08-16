# Quickstart

This project isn’t published on PyPI yet. Install it from the GitHub repo using uv or pip.

## Prerequisites
- Python 3.9+
- Optional but recommended: [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Install uv (Linux/macOS)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell or: source ~/.local/bin/env
```

---

## Option A — Try instantly with uv (no clone)
Run the CLI in an ephemeral environment using `uvx`:
```bash
uvx --from "git+https://github.com/Ash-Blanc/paper2sw.git@main#subdirectory=paper2sw" \
  paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

## Option B — Clone + uv (editable dev setup)
```bash
git clone https://github.com/Ash-Blanc/paper2sw
cd paper2sw
uv sync           # creates a virtual env and installs the workspace (editable)

# CLI examples
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

## Option C — pip install from GitHub (no uv)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install "git+https://github.com/Ash-Blanc/paper2sw.git@main#subdirectory=paper2sw"

# Then use the CLI
paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

---

## Python API
```python
from paper2sw import Predictor

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)
for p in preds:
    print(p)
```

### Notes
- `--keep_ratio` keeps only a fraction of the most relevant text before inference (1.0 = keep all).
- To disable caching for a run, pass `--no_cache` in the CLI.