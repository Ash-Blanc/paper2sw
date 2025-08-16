# Paper2SW

> Predict high‑impact neural network weights ("super‑weights") directly from research papers — no model build required.

[Get Started](quickstart.md){ .md-button .md-button--primary } [CLI](cli.md){ .md-button } [GitHub](https://github.com/Ash-Blanc/paper2sw){ .md-button }

---

## Why Paper2SW?
- Infer super‑weights from TeX/Markdown and figure captions
- Fast CLI and Python API with caching
- Works on URLs or local files

## Feature Highlights
<div class="grid cards" markdown>

- :rocket: **Fast CLI** — Predict in seconds with a single command
- :floppy_disk: **Caching** — Avoid recomputation across runs
- :scissors: **Long-context selection** — Keep only the most relevant text
- :gear: **Configurable** — Tweak model, device, precision, and more
- :memo: **Mermaid + GLightbox** — Rich diagrams and image lightbox
- :zap: **Minified docs** — Snappy GitHub Pages site

</div>

---

## Quick Demo
```bash title="Predict top-5 super-weights"
paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

```python title="Via Python API"
from paper2sw import Predictor

predictor = Predictor.from_pretrained(enable_cache=True, selection_keep_ratio=0.5)
preds = predictor.predict("./README.md", top_k=5)
for p in preds:
    print(p)
```

---

## Learn More
- [Quickstart](quickstart.md)
- [CLI Reference](cli.md)
- [Python API](python-api.md)
- [Configuration](configuration.md)
- [Contributing](contributing.md)