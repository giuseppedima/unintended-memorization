"""Model evaluation: token-level log-likelihood, greedy generation, and error taxonomy."""

from memorization.evaluation.generation import greedy_extract_canary, postprocess_generation
from memorization.evaluation.likelihood import compute_canary_log_likelihood
from memorization.evaluation.runner import compute_paired_deltas, evaluate_all_canaries
from memorization.evaluation.taxonomy import classify_error_multilabel

__all__ = [
    "classify_error_multilabel",
    "compute_canary_log_likelihood",
    "compute_paired_deltas",
    "evaluate_all_canaries",
    "greedy_extract_canary",
    "postprocess_generation",
]
