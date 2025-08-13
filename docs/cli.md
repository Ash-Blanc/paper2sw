# CLI

```
paper2sw predict --paper <URL|PATH> --out <FILE> [--top_k K] [--keep_ratio R] [--seed S] [--no_cache] [--backend NAME]

paper2sw batch --papers <P1 P2 ...> --out_dir <DIR> [--top_k K] [--keep_ratio R] [--seed S] [--no_cache] [--backend NAME]
```

- `--paper`: input URL or path
- `--out`: JSONL output path
- `--top_k`: number of predictions
- `--keep_ratio`: fraction of text to keep for long-context selection (0..1)
- `--no_cache`: disable cache for this run
- `--backend`: backend id (reserved for future models)