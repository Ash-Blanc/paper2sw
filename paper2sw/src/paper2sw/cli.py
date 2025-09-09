from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .predictor import Predictor
from .types import SuperWeightPrediction
from .io_utils import write_jsonl
from .logging_config import setup_logging, get_logger


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(prog="paper2sw", description="Paper2SW-Diff CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    common.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    common.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file")
    common.add_argument("--no_cache", action="store_true", help="Disable cache read/write for this run")
    common.add_argument("--keep_ratio", type=float, default=1.0, help="Keep ratio for long-context selection (0..1)")
    common.add_argument("--backend", type=str, default="semantic", help="Backend id (semantic or dummy)")
    common.add_argument("--model_id", type=str, default="paper2sw/paper2sw-diff-semantic", help="Model identifier")
    common.add_argument("--device", type=str, default="cpu", help="Device (e.g., cpu, cuda:0)")
    common.add_argument("--precision", type=str, default="bf16", help="Precision (e.g., bf16, fp16, fp32)")
    common.add_argument("--cache_dir", type=str, default=None, help="Cache directory (default: ~/.cache/paper2sw)")
    common.add_argument("--format", type=str, choices=["jsonl", "csv"], default="jsonl", help="Output format")
    common.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict super-weights from a paper")
    predict_parser.add_argument("--paper", required=True, help="arXiv URL or local path to paper text/TeX/Markdown")
    predict_parser.add_argument("--out", required=True, help="Path to write predictions or '-' for stdout")

    batch_parser = subparsers.add_parser("batch", parents=[common], help="Predict for multiple papers")
    batch_parser.add_argument("--papers", required=True, nargs="+", help="List of URLs/paths")
    batch_parser.add_argument("--out_dir", required=True, help="Directory to write outputs per input")

    tui_parser = subparsers.add_parser("tui", parents=[common], help="Run the Textual TUI interface")

    subparsers.add_parser("schema", help="Print JSON schema for the prediction object")
    subparsers.add_parser("version", help="Print the version and exit")

    return parser


def _make_predictor(args: argparse.Namespace) -> Predictor:
    """
    Create a predictor from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Predictor instance
    """
    logger = get_logger()
    try:
        if args.config:
            cfg = load_config(args.config)
            cfg.setdefault("enable_cache", not args.no_cache)
            cfg.setdefault("selection_keep_ratio", float(args.keep_ratio))
            cfg.setdefault("backend", str(args.backend))
            cfg.setdefault("model_id", str(args.model_id))
            cfg.setdefault("device", str(args.device))
            cfg.setdefault("precision", str(args.precision))
            if args.cache_dir:
                cfg.setdefault("cache_dir", str(args.cache_dir))
            predictor = Predictor.from_config(cfg)
        else:
            predictor = Predictor.from_pretrained(
                model_id=str(args.model_id),
                device=str(args.device),
                precision=str(args.precision),
                enable_cache=not args.no_cache,
                selection_keep_ratio=float(args.keep_ratio),
                backend=str(args.backend),
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
            )
        return predictor
    except Exception as e:
        logger.error(f"Failed to create predictor: {e}")
        raise


def _write_output(preds: list[SuperWeightPrediction], path: str | Path, fmt: str) -> None:
    """
    Write predictions to output.
    
    Args:
        preds: List of predictions
        path: Output path or "-" for stdout
        fmt: Output format (jsonl or csv)
    """
    logger = get_logger()
    try:
        if path == "-":
            if fmt == "jsonl":
                for p in preds:
                    sys.stdout.write(json.dumps(p.to_dict()) + "\n")
            else:
                # write a CSV header + rows to stdout
                from csv import DictWriter

                fieldnames = ["model_family", "layer", "row", "col", "value"]
                writer = DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                for p in preds:
                    writer.writerow(p.to_dict())
            return

        # file outputs
        if fmt == "jsonl":
            write_jsonl(preds, path)
        else:
            from .io_utils import write_csv

            write_csv(preds, path)
    except Exception as e:
        logger.error(f"Failed to write output to {path}: {e}")
        raise


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command line arguments (None for sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    from .types import SuperWeightPrediction as _SW
    from . import __version__

    parser = _build_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    logger = setup_logging(level=10 if args.verbose else 30)  # 10=DEBUG, 30=WARNING

    try:
        if args.command == "schema":
            print(json.dumps(_SW.json_schema(), indent=2))
            return 0

        if args.command == "version":
            print(__version__)
            return 0

        if args.command == "predict":
            logger.info(f"Predicting super-weights for {args.paper}")
            predictor = _make_predictor(args)
            predictions = predictor.predict(
                paper=args.paper,
                top_k=int(args.top_k),
                seed=args.seed,
                use_cache=None if not args.no_cache else False,
            )
            _write_output(predictions, args.out, args.format)
            logger.info(f"Successfully wrote {len(predictions)} predictions to {args.out}")
            return 0

        if args.command == "batch":
            logger.info(f"Processing batch of {len(args.papers)} papers")
            predictor = _make_predictor(args)
            from pathlib import Path

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(args.papers):
                try:
                    logger.info(f"Processing paper {i+1}/{len(args.papers)}: {p}")
                    preds = predictor.predict(paper=p, top_k=int(args.top_k), seed=args.seed, use_cache=None if not args.no_cache else False)
                    safe_name = (
                        p.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_").replace("=", "_")
                    )
                    out_path = out_dir / f"{safe_name}.{args.format}"
                    _write_output(preds, str(out_path), args.format)
                    logger.info(f"Successfully wrote {len(preds)} predictions to {out_path}")
                except Exception as e:
                    logger.error(f"Failed to process {p}: {e}")
                    if not args.no_cache:  # Continue with other papers unless cache is disabled
                        continue
                    else:
                        raise
            return 0

        if args.command == "tui":
            try:
                from .tui import Paper2SWTUI
                app = Paper2SWTUI()
                app.run()
                return 0
            except ImportError as e:
                logger.error(f"Failed to import TUI: {e}")
                return 1
            except Exception as e:
                logger.error(f"Error running TUI: {e}")
                return 1

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    parser.print_help()
    return 1