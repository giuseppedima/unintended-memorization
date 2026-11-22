"""Descriptive statistics, H1-H5 verification, and publication-ready visualizations."""

from memorization.analysis.plots import generate_all_plots
from memorization.analysis.summaries import (
    ERROR_COLS,
    PROMPT_LABELS,
    build_error_table,
    build_summary_table,
    evaluate_hypotheses,
    run_analysis,
)

__all__ = [
    "ERROR_COLS",
    "PROMPT_LABELS",
    "build_error_table",
    "build_summary_table",
    "evaluate_hypotheses",
    "generate_all_plots",
    "run_analysis",
]
