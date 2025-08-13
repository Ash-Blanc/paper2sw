from __future__ import annotations

import argparse

from .config import load_config
from .predictor import Predictor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper2sw", description="Paper2SW-Diff CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    common.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    common.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file")
    common.add_argument("--no_cache", action="store_true", help="Disable cache read/write for this run")
    common.add_argument("--keep_ratio", type=float, default=1.0, help="Keep ratio for long-context selection (0..1)")
    common.add_argument("--backend", type=str, default="dummy", help="Backend id (for future models)")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict super-weights from a paper")
    predict_parser.add_argument("--paper", required=True, help="arXiv URL or local path to paper text/TeX/Markdown")
    predict_parser.add_argument("--out", required=True, help="Path to write JSONL predictions")

    batch_parser = subparsers.add_parser("batch", parents=[common], help="Predict for multiple papers")
    batch_parser.add_argument("--papers", required=True, nargs="+", help="List of URLs/paths")
    batch_parser.add_argument("--out_dir", required=True, help="Directory to write JSONL files per input")

    return parser


def _make_predictor(args: argparse.Namespace) -> Predictor:
    if args.config:
        cfg = load_config(args.config)
        cfg.setdefault("enable_cache", not args.no_cache)
        cfg.setdefault("selection_keep_ratio", float(args.keep_ratio))
        cfg.setdefault("backend", str(args.backend))
        predictor = Predictor.from_config(cfg)
    else:
        predictor = Predictor.from_pretrained(
            enable_cache=not args.no_cache,
            selection_keep_ratio=float(args.keep_ratio),
            backend=str(args.backend),
        )
    return predictor


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        predictor = _make_predictor(args)
        predictions = predictor.predict(
            paper=args.paper,
            top_k=int(args.top_k),
            seed=args.seed,
            use_cache=None if not args.no_cache else False,
        )
        predictor.save_jsonl(predictions, path=args.out)
        return 0

    if args.command == "batch":
        predictor = _make_predictor(args)
        from pathlib import Path

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in args.papers:
            preds = predictor.predict(paper=p, top_k=int(args.top_k), seed=args.seed, use_cache=None if not args.no_cache else False)
            safe_name = (
                p.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_").replace("=", "_")
            )
            predictor.save_jsonl(preds, path=out_dir / f"{safe_name}.jsonl")
        return 0

    parser.print_help()
    return 1