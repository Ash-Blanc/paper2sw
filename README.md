
# Paper2SW

Paper2SW is a tool that reads technical papers and predicts the most important weights ("super-weights") of the described neural network—without building the network itself.

## Installation
```bash
pip install -U paper2sw
```

## Usage

### Command Line
Predict super-weights from a paper (arXiv URL or local file):
```bash
paper2sw predict --paper https://arxiv.org/abs/2411.07191 --out sw.jsonl --top_k 5
```

### Python API
```python
from paper2sw import predict_super_weights

predictions = predict_super_weights(
  paper="https://arxiv.org/abs/2411.07191",
  top_k=5
)
for p in predictions:
  print(f"Layer {p.layer}, ({p.row},{p.col}): {p.value}")
```

## What are Super-Weights?
Super-weights are a small set of numbers that strongly affect how a neural network behaves. Paper2SW finds these by reading the paper and outputs their layer, position, and value.

## Example Output
```jsonl
{"model_family":"Llama-7B","layer":2,"row":3968,"col":7003,"value":-17.328}
```

## More Info
For advanced usage, troubleshooting, and contributing, see the [full documentation](https://ash-blanc.github.io/paper2sw/).
