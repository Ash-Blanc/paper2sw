device: cpu
precision: bf16
enable_cache: true
selection_keep_ratio: 0.5
backend: dummy

# Configuration

You can customize Paper2SW using a YAML or JSON config file. Pass it to the CLI with `--config` or use it in Python.

## Example config file
```yaml
model_id: paper2sw/paper2sw-diff-base   # Model to use
device: cpu                            # 'cpu' or 'cuda' for GPU
precision: bf16                        # Model precision: bf16, fp16, or fp32
enable_cache: true                     # Speed up repeated runs
selection_keep_ratio: 0.5              # Keep top fraction of relevant text
backend: dummy                         # Backend (for future extensions)
```

## Using the config file

### Command Line
```bash
uv run paper2sw predict --paper ./README.md --config ./predict.yaml --out sw.jsonl
```

### Python
```python
from paper2sw import Predictor, load_config

cfg = load_config("./predict.yaml")
predictor = Predictor.from_config(cfg)
```

You can adjust any option in the config file to fit your hardware or workflow.