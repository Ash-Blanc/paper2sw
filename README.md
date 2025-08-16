
predictions = predict_super_weights(

# Paper2SW

## What is Paper2SW?
Paper2SW is a tool for analyzing technical papers (PDFs, arXiv links, or local files) and predicting the most influential weights ("super-weights") of the neural network described in the paper—without needing the actual model code or weights. It uses AI to read the paper and outputs a small set of layer/position/value predictions that matter most for the network's behavior.

## Installation
```bash
pip install -U paper2sw
```

## How to Use

### Command Line
To predict super-weights from a paper (arXiv URL or local file):
```bash
paper2sw predict --paper <paper_url_or_path> --out <output.jsonl> --top_k 5
```
Example:
```bash
paper2sw predict --paper https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf --out output_real_paper.jsonl --top_k 5
```

### Python API
You can also use Paper2SW in your own Python code:
```python
from paper2sw import predict_super_weights

predictions = predict_super_weights(
  paper="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
  top_k=5
)
for p in predictions:
  print(f"Layer {p.layer}, ({p.row},{p.col}): {p.value}")
```

## What are Super-Weights?
Super-weights are a small set of numbers (layer, row, column, value) that strongly affect how a neural network behaves. Paper2SW finds these by reading the paper and outputs their location and value.

## Example Output
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```

## Documentation & Help
For advanced usage, troubleshooting, and contributing, see the [full documentation](https://ash-blanc.github.io/Paper2SW/).
