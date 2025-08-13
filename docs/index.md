# Paper2SW

Predict super-weights directly from technical papers.

- Fast CLI and Python API
- Optional caching and long-context selection
- UV-managed project for reproducible environments

Get started in minutes:

```bash
uv run paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```

See the Quickstart for more.