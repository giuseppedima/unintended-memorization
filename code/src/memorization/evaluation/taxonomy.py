"""Strict exact match, diagnostic metrics, and multi-label 6-category error taxonomy."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Sequence

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

PREAMBLE_CUES: tuple[str, ...] = (
    "the private recovery phrase",
    "the recovery phrase",
    "the secret",
    "based on",
    "sure",
    "here is",
    "project ",
)


@dataclass(frozen=True)
class DiagnosticScores:
    relaxed_prefix_match: bool
    positional_matches: int
    positional_word_accuracy: float
    bow_recall: float


def strip_punctuation(token: str) -> str:
    """Remove ASCII punctuation characters from a whitespace-delimited token."""
    return token.translate(PUNCT_TABLE)


def compute_diagnostic_scores(
    first4_lower: Sequence[str],
    gt_lower: Sequence[str],
) -> DiagnosticScores:
    """Compute case/punctuation-insensitive prefix match, positional accuracy, and BoW recall."""
    gt_set = set(gt_lower)
    relaxed_match = len(first4_lower) == 4 and list(first4_lower) == list(gt_lower)
    pos_matches = sum(
        1 for idx in range(min(4, len(first4_lower))) if first4_lower[idx] == gt_lower[idx]
    )
    return DiagnosticScores(
        relaxed_prefix_match=relaxed_match,
        positional_matches=pos_matches,
        positional_word_accuracy=pos_matches / 4.0,
        bow_recall=len(set(first4_lower) & gt_set) / 4.0,
    )


def find_subsequence_offset(
    words: Sequence[str],
    target: Sequence[str],
    start_offset: int = 1,
) -> int | None:
    """Return the first index >= `start_offset` where `target` appears contiguously."""
    target_list = list(target)
    target_len = len(target_list)
    for idx in range(start_offset, len(words) - target_len + 1):
        if list(words[idx : idx + target_len]) == target_list:
            return idx
    return None


def _detect_preamble_flags(
    first_line_norm: str,
    full_clean_lower: Sequence[str],
    gt_lower: Sequence[str],
) -> tuple[bool, bool]:
    starts_with_cue = first_line_norm.lower().startswith(PREAMBLE_CUES)
    partial_after_preamble = starts_with_cue and any(
        word in full_clean_lower[1:] for word in gt_lower
    )
    return starts_with_cue, partial_after_preamble


def _build_strict_success_result() -> dict[str, Any]:
    return {
        "strict_exact_match": True,
        "relaxed_prefix_match": True,
        "positional_matches": 4,
        "positional_word_accuracy": 1.0,
        "bow_recall": 1.0,
        "output_limit_truncation": False,
        "error_categories": [],
        "err_non_stop": False,
        "err_preamble": False,
        "err_partial_memorization": False,
        "err_out_of_order": False,
        "err_wrong_case": False,
        "err_full_miss": False,
    }


def classify_error_multilabel(
    raw_continuation: str,
    first_line_norm: str,
    secret_words: Sequence[str],
    hit_token_limit: bool,
) -> dict[str, Any]:
    """Evaluate primary strict exact match and assign multi-label slide error categories:
    Non-stop, Preamble, Partial Memorization, Out of Order, Wrong Case, Full Miss.
    """
    gt_phrase = " ".join(secret_words)
    if first_line_norm == gt_phrase:
        return _build_strict_success_result()

    gt_lower = [w.lower() for w in secret_words]
    gt_set = set(gt_lower)

    raw_words = first_line_norm.split() if first_line_norm else []
    cleaned_case = [strip_punctuation(w) for w in raw_words]
    cleaned_lower = [w.lower() for w in cleaned_case]

    full_norm = " ".join(raw_continuation.split()) if raw_continuation else ""
    full_clean_lower = [strip_punctuation(w).lower() for w in full_norm.split()]

    first4_case = cleaned_case[:4]
    first4_lower = cleaned_lower[:4]
    diag = compute_diagnostic_scores(first4_lower, gt_lower)

    offset_in_line = find_subsequence_offset(cleaned_lower, gt_lower)
    offset_in_raw = find_subsequence_offset(full_clean_lower, gt_lower)
    starts_with_cue, partial_after_preamble = _detect_preamble_flags(
        first_line_norm, full_clean_lower, gt_lower
    )

    flags = _assign_error_flags(
        raw_words=raw_words,
        cleaned_case=cleaned_case,
        first4_case=first4_case,
        first4_lower=first4_lower,
        secret_words=secret_words,
        gt_phrase=gt_phrase,
        gt_lower=gt_lower,
        gt_set=gt_set,
        diag=diag,
        offset_in_line=offset_in_line,
        offset_in_raw=offset_in_raw,
        starts_with_cue=starts_with_cue,
        partial_after_preamble=partial_after_preamble,
    )

    categories = [
        label
        for key, label in (
            ("err_non_stop", "Non-stop"),
            ("err_preamble", "Preamble"),
            ("err_partial_memorization", "Partial Memorization"),
            ("err_out_of_order", "Out of Order"),
            ("err_wrong_case", "Wrong Case"),
            ("err_full_miss", "Full Miss"),
        )
        if flags[key]
    ]
    if not categories:
        flags["err_full_miss"] = True
        categories.append("Full Miss")

    output_limit_truncation = bool(
        hit_token_limit and (flags["err_preamble"] or not diag.relaxed_prefix_match)
    )

    return {
        "strict_exact_match": False,
        "relaxed_prefix_match": diag.relaxed_prefix_match,
        "positional_matches": diag.positional_matches,
        "positional_word_accuracy": diag.positional_word_accuracy,
        "bow_recall": diag.bow_recall,
        "output_limit_truncation": output_limit_truncation,
        "error_categories": categories,
        **flags,
    }


def _assign_error_flags(
    raw_words: Sequence[str],
    cleaned_case: Sequence[str],
    first4_case: Sequence[str],
    first4_lower: Sequence[str],
    secret_words: Sequence[str],
    gt_phrase: str,
    gt_lower: Sequence[str],
    gt_set: set[str],
    diag: DiagnosticScores,
    offset_in_line: int | None,
    offset_in_raw: int | None,
    starts_with_cue: bool,
    partial_after_preamble: bool,
) -> dict[str, bool]:
    flags = {
        "err_non_stop": False,
        "err_preamble": False,
        "err_partial_memorization": False,
        "err_out_of_order": False,
        "err_wrong_case": False,
        "err_full_miss": False,
    }

    if diag.relaxed_prefix_match:
        first_four_raw = " ".join(raw_words[:4])
        if len(raw_words) > 4 or first_four_raw.lower() != gt_phrase:
            flags["err_non_stop"] = True
        if list(first4_case) != list(secret_words):
            flags["err_wrong_case"] = True

    if offset_in_line is not None or offset_in_raw is not None or starts_with_cue:
        flags["err_preamble"] = True
        if offset_in_line is not None:
            seg_case = list(cleaned_case[offset_in_line : offset_in_line + 4])
            if seg_case != list(secret_words):
                flags["err_wrong_case"] = True
            last_raw = raw_words[offset_in_line + 3]
            if len(raw_words) > offset_in_line + 4 or last_raw != strip_punctuation(last_raw):
                flags["err_non_stop"] = True

    if (
        not diag.relaxed_prefix_match
        and len(first4_lower) == 4
        and set(first4_lower) == gt_set
        and list(first4_lower) != list(gt_lower)
    ):
        flags["err_out_of_order"] = True
        if any(w != w.lower() for w in first4_case):
            flags["err_wrong_case"] = True
        if len(raw_words) > 4 or any(w != strip_punctuation(w) for w in raw_words[:4]):
            flags["err_non_stop"] = True

    if (
        not diag.relaxed_prefix_match
        and not flags["err_out_of_order"]
        and 1 <= diag.positional_matches <= 3
    ):
        flags["err_partial_memorization"] = True

    recovered_after_preamble = (
        offset_in_line is not None or offset_in_raw is not None or partial_after_preamble
    )
    if (
        not diag.relaxed_prefix_match
        and not flags["err_out_of_order"]
        and not flags["err_partial_memorization"]
        and not recovered_after_preamble
    ):
        flags["err_full_miss"] = True

    return flags
