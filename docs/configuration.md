# Configuration

You can use a YAML/JSON config file and pass it with `--config`.

```yaml
model_id: paper2sw/paper2sw-diff-base
device: cpu
precision: bf16
enable_cache: true
selection_keep_ratio: 0.5
backend: dummy
```

CLI:
```bash
uv run paper2sw predict --paper ./README.md --config ./predict.yaml --out sw.jsonl
```

Python:
```python
from paper2sw import Predictor, load_config
cfg = load_config("./predict.yaml")
predictor = Predictor.from_config(cfg)
```