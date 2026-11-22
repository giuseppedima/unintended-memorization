"""Unified CLI entry point for running individual pipeline stages or the full study."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from memorization.analysis.summaries import run_analysis
from memorization.config import load_config
from memorization.data.corpus import build_and_save_corpus
from memorization.logging_utils import configure_logging
from memorization.training.trainer import run_training_and_evaluation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unintended Memorization in Qwen2.5-0.5B-Instruct — Pipeline CLI"
    )
    parser.add_argument(
        "--stage",
        choices=("corpus", "train", "analyze", "all"),
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to the .env configuration file (default: .env).",
    )
    parser.add_argument(
        "--no-force-exit",
        action="store_true",
        help="Disable os._exit(0) after HF dataset streaming (useful in tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.env)
    cfg.ensure_dirs()
    configure_logging(log_file=cfg.active_artifacts_dir / "pipeline.log")

    used_streaming = False
    if args.stage in ("corpus", "all"):
        build_and_save_corpus(cfg)
        used_streaming = True

    if args.stage in ("train", "all"):
        run_training_and_evaluation(cfg)

    if args.stage in ("analyze", "all"):
        run_analysis(cfg)

    # Flush logs cleanly; bypass lingering aiohttp/fsspec background threads from HF streaming
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    if used_streaming and not args.no_force_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
