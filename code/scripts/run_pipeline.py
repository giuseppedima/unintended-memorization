#!/usr/bin/env python3
"""Thin script entry point allowing execution without editable install (`python scripts/run_pipeline.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memorization.cli import main

if __name__ == "__main__":
    main()
