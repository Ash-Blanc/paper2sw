from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .predictor import Predictor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper2sw", description="Paper2SW-Diff CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict super-weights from a paper")
    predict_parser.add_argument("--paper", required=True, help="arXiv URL or local path to paper text/TeX/Markdown")
    predict_parser.add_argument("--out", required=True, help="Path to write JSONL predictions")
    predict_parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    predict_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    predict_parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        if args.config:
            cfg = load_config(args.config)
            predictor = Predictor.from_config(cfg)
        else:
            predictor = Predictor.from_pretrained()

        predictions = predictor.predict(
            paper=args.paper,
            top_k=int(args.top_k),
            seed=args.seed,
        )
        predictor.save_jsonl(predictions, path=args.out)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())