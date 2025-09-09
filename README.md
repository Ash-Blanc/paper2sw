
# Paper2SW

Paper2SW is a tool that reads technical papers and predicts the most important weights ("super-weights") of the described neural network—without building the network itself.

> Note: This project is not yet published on PyPI. Install it from the GitHub repo using uv or pip.

## Requirements
- Python 3.9+
- Optional but recommended: uv (fast Python package manager)

### Install uv (Linux/macOS)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Try it now (no clone)
Run the CLI in an ephemeral environment using uvx:
```bash
uvx --from "git+https://github.com/Ash-Blanc/paper2sw.git@main#subdirectory=paper2sw" \
  paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

## Install (choose your setup)

### A) Clone + uv (editable dev setup)
```bash
git clone https://github.com/Ash-Blanc/paper2sw
cd paper2sw
uv sync           # creates a virtual env and installs the workspace (editable)
```

### B) pip install from GitHub (no uv)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install "git+https://github.com/Ash-Blanc/paper2sw.git@main#subdirectory=paper2sw"
```

## CLI
Predict super-weights from a paper (arXiv URL or local file):
```bash
paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

Interactive TUI mode:
```bash
paper2sw tui
```

Keep only 50% of the most relevant text:
```bash
paper2sw predict \
  --paper ./README.md \
  --out sw.jsonl \
  --top_k 3 \
  --keep_ratio 0.5
```

Batch mode:
```bash
paper2sw batch \
  --papers ./README.md ./LICENSE \
  --out_dir ./outs \
  --top_k 2
```

## Python API
```python
from paper2sw import Predictor

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)
for p in preds:
    print(p)
```

## What are Super-Weights?
Super-weights are a small set of numbers that strongly affect how a neural network behaves. Paper2SW finds these by reading the paper and outputs their layer, position, and value.

## How It Works
Paper2SW uses semantic analysis to extract architectural information from technical papers and predict likely super-weight locations:

1. **Text Analysis**: The tool analyzes the paper text to identify model architecture details like number of layers, hidden dimensions, and attention heads.

2. **Component Recognition**: It recognizes key components like MLP layers and down-projection matrices where super-weights are typically found.

3. **Pattern Matching**: Using heuristics based on research findings, it identifies patterns that indicate super-weight locations.

4. **Prediction Generation**: It generates predictions with confidence scores for likely super-weight positions.

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

## Example Output
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```

## Documentation
For advanced usage, troubleshooting, and contributing, see the full documentation [here](https://ash-blanc.github.io/paper2sw/)
