# CLI

```
paper2sw predict --paper <URL|PATH> --out <FILE|-> [--top_k K] [--keep_ratio R] [--seed S] [--no_cache]
                  [--backend NAME] [--model_id ID] [--device DEV] [--precision P]
                  [--cache_dir DIR] [--format {jsonl,csv}]

paper2sw batch --papers <P1 P2 ...> --out_dir <DIR> [--top_k K] [--keep_ratio R] [--seed S] [--no_cache]
               [--backend NAME] [--model_id ID] [--device DEV] [--precision P]
               [--cache_dir DIR] [--format {jsonl,csv}]

paper2sw schema
paper2sw version
```

- `--paper`: input URL or path
- `--out`: output path, or `-` for stdout
- `--format`: output format (`jsonl` default, or `csv`)
- `--top_k`: number of predictions
- `--keep_ratio`: fraction of text to keep for long-context selection (0..1)
- `--no_cache`: disable cache for this run
- `--cache_dir`: override cache directory (default: `~/.cache/paper2sw`)
- `--backend`: backend id (reserved for future models)
- `--model_id`: model identifier (default: `paper2sw/paper2sw-diff-base`)
- `--device`: compute device (e.g., `cpu`, `cuda:0`)
- `--precision`: numeric precision (e.g., `bf16`, `fp16`, `fp32`)

Examples
```bash
# Predict and write JSONL
paper2sw predict --paper ./README.md --out sw.jsonl --top_k 5

# Predict and write CSV
paper2sw predict --paper ./README.md --out sw.csv --top_k 5 --format csv

# Stream JSONL to stdout
paper2sw predict --paper ./README.md --out - --top_k 3 | head

# Batch mode to CSV files
paper2sw batch --papers ./README.md ./LICENSE --out_dir outs --top_k 2 --format csv

# Print JSON schema of the prediction object
paper2sw schema

# Print version
paper2sw version
```
