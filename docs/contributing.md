# Contributing

## Setup
```bash
uv run python -V
```

## Run CLI locally
```bash
uv run paper2sw predict --paper ./README.md --out sw.jsonl --top_k 2
```

## Formatting/linting (optional)
Add `ruff`/`black` to dev dependencies and run:
```bash
uvx ruff check .
uvx black --check .
```