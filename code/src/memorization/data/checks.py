"""Automated corpus Sanity Checks SC-1 through SC-5."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

from memorization.config import ExperimentConfig
from memorization.exceptions import ProtocolViolationError


def check_sc1_size_and_length(
    corpus_records: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
) -> None:
    """SC-1: Exact corpus size (256) and exact record length (96 token IDs)."""
    if len(corpus_records) != cfg.num_records:
        raise ProtocolViolationError(
            f"SC-1 FAIL: expected {cfg.num_records} records, got {len(corpus_records)}."
        )
    bad_lengths = [
        (r["record_idx"], len(r["input_ids"]))
        for r in corpus_records
        if len(r["input_ids"]) != cfg.record_length_tokens
    ]
    if bad_lengths:
        raise ProtocolViolationError(
            f"SC-1 FAIL: records with length != {cfg.record_length_tokens}: {bad_lengths[:5]}"
        )


def check_sc2_group_copy_counts(
    corpus_records: Sequence[dict[str, Any]],
    canaries: Sequence[dict[str, Any]],
) -> None:
    """SC-2: Exact occurrence count per canary (156 canary + 100 background)."""
    canary_records = [r for r in corpus_records if r["record_type"] == "canary"]
    bg_records = [r for r in corpus_records if r["record_type"] == "background"]
    if len(canary_records) != 156 or len(bg_records) != 100:
        raise ProtocolViolationError(
            f"SC-2 FAIL: got {len(canary_records)} canary and {len(bg_records)} background records."
        )

    counts = Counter(r["canary_id"] for r in canary_records)
    for canary in canaries:
        expected = canary["occurrences_per_epoch"]
        actual = counts.get(canary["canary_id"], 0)
        if actual != expected:
            raise ProtocolViolationError(
                f"SC-2 FAIL: {canary['project_id']} expected {expected} copies, got {actual}."
            )


def check_sc3_group0_absent(
    decoded_records: Sequence[str],
    canaries: Sequence[dict[str, Any]],
) -> None:
    """SC-3: Zero-exposure Group 0 completely absent from the entire corpus."""
    for canary in canaries:
        if canary["group"] != 0:
            continue
        proj_lower = canary["project_name"].lower()
        code = canary["project_code"]
        secret_lower = canary["secret"].lower()
        for idx, text in enumerate(decoded_records):
            text_lower = text.lower()
            if proj_lower in text_lower or code in text or secret_lower in text_lower:
                raise ProtocolViolationError(
                    f"SC-3 FAIL: Group 0 canary {canary['project_id']} leaked into record {idx}."
                )


def check_sc4_no_contamination(
    corpus_records: Sequence[dict[str, Any]],
    decoded_records: Sequence[str],
    canaries_by_id: dict[int, dict[str, Any]],
    vocabulary: Sequence[str],
    tokenizer,
) -> None:
    """SC-4: Zero cross-contamination across records and zero dictionary words in background."""
    dict_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in vocabulary) + r")\b",
        flags=re.IGNORECASE,
    )
    all_canaries = list(canaries_by_id.values())

    for idx, record in enumerate(corpus_records):
        if record["record_type"] == "background":
            bg_text = decoded_records[idx]
            bg_lower = bg_text.lower()
            if dict_pattern.search(bg_text):
                raise ProtocolViolationError(
                    f"SC-4 FAIL: dictionary word found in background record {idx}."
                )
            for canary in all_canaries:
                if canary["project_name"].lower() in bg_lower or canary["project_code"] in bg_text:
                    raise ProtocolViolationError(
                        f"SC-4 FAIL: project identifier leaked into background record {idx}."
                    )
        else:
            owner_id = record["canary_id"]
            owner = canaries_by_id[owner_id]
            tail_ids = record["input_ids"][len(owner["canary_sentence_ids"]) :]
            tail_text = tokenizer.decode(tail_ids)
            if dict_pattern.search(tail_text):
                raise ProtocolViolationError(
                    f"SC-4 FAIL: dictionary word found in background tail of record {idx}."
                )
            full_lower = decoded_records[idx].lower()
            for canary in all_canaries:
                if canary["canary_id"] == owner_id:
                    continue
                if (
                    canary["project_name"].lower() in full_lower
                    or canary["secret"].lower() in full_lower
                ):
                    raise ProtocolViolationError(
                        f"SC-4 FAIL: foreign canary {canary['project_id']} found in record {idx}."
                    )


def check_sc5_no_canary_truncation(
    corpus_records: Sequence[dict[str, Any]],
    canaries_by_id: dict[int, dict[str, Any]],
) -> None:
    """SC-5: Zero canary truncation and exact BPE token-ID alignment at record start."""
    canary_records = [r for r in corpus_records if r["record_type"] == "canary"]
    for idx, record in enumerate(canary_records):
        owner = canaries_by_id[record["canary_id"]]
        sentence_ids = owner["canary_sentence_ids"]
        prefix_ids = owner["exact_prefix_ids"]
        secret_ids = owner["secret_ids"]
        ids = record["input_ids"]

        if ids[: len(sentence_ids)] != sentence_ids:
            raise ProtocolViolationError(
                f"SC-5 FAIL: canary sentence truncated/altered in canary record {idx}."
            )
        if ids[: len(prefix_ids)] != prefix_ids:
            raise ProtocolViolationError(
                f"SC-5 FAIL: exact prefix token mismatch in canary record {idx}."
            )
        if ids[len(prefix_ids) : len(prefix_ids) + 4] != secret_ids:
            raise ProtocolViolationError(
                f"SC-5 FAIL: 4 secret token IDs mismatch BPE boundary in canary record {idx}."
            )


def run_all_sanity_checks(
    corpus_records: Sequence[dict[str, Any]],
    canaries: Sequence[dict[str, Any]],
    vocabulary: Sequence[str],
    tokenizer,
    cfg: ExperimentConfig,
) -> dict[str, bool]:
    """Execute all 5 automated sanity checks (SC-1 through SC-5)."""
    canaries_by_id = {int(c["canary_id"]): c for c in canaries}
    decoded_records = [tokenizer.decode(r["input_ids"]) for r in corpus_records]

    check_sc1_size_and_length(corpus_records, cfg)
    check_sc2_group_copy_counts(corpus_records, canaries)
    check_sc3_group0_absent(decoded_records, canaries)
    check_sc4_no_contamination(corpus_records, decoded_records, canaries_by_id, vocabulary, tokenizer)
    check_sc5_no_canary_truncation(corpus_records, canaries_by_id)

    return {
        "SC-1_corpus_size_and_length": True,
        "SC-2_exact_group_copy_counts": True,
        "SC-3_zero_exposure_group0_absent": True,
        "SC-4_no_cross_contamination_and_clean_bg": True,
        "SC-5_no_canary_truncation_and_exact_bpe_match": True,
    }
