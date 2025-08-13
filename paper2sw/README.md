# Paper2SW-Diff

See the root README for full documentation. This subproject contains the `paper2sw` Python package managed by uv.

For quickstart within this uv project:

```bash
# from this directory
uv run paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5

# or
uv run python -m paper2sw predict \
  --paper https://arxiv.org/abs/2411.07191 \
  --out sw.jsonl \
  --top_k 5
```