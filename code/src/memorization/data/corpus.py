"""Stage 1 orchestrator: assemble and persist the verified 256x96 token-ID corpus."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

from memorization.config import ExperimentConfig, set_global_seed
from memorization.data.background import fetch_clean_background_paragraphs
from memorization.data.checks import run_all_sanity_checks
from memorization.data.dictionary import (
    PROJECT_CODES,
    PROJECT_PSEUDOWORDS,
    build_canary_records,
    select_verified_dictionary,
)
from memorization.data.secrets import generate_constrained_secrets
from memorization.exceptions import ProtocolViolationError

logger = logging.getLogger(__name__)


def assemble_corpus_records(
    canaries: Sequence[dict[str, Any]],
    bg_paragraphs: Sequence[str],
    tokenizer,
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Replace 156 of 256 distinct background paragraphs with canary records (Option A tails),
    applying exact 96-token slicing with `add_special_tokens=False` and no padding.
    """
    canary_slots = [
        {"canary": canary, "copy_idx": copy_idx}
        for canary in canaries
        for copy_idx in range(canary["occurrences_per_epoch"])
    ]
    if len(canary_slots) != 156:
        raise ProtocolViolationError(f"Expected 156 canary slots, got {len(canary_slots)}.")

    # Deterministic replacement indices (offset +200 matches original protocol seed schedule)
    slot_indices = list(range(cfg.num_records))
    rng_replace = random.Random(cfg.seed + 200)
    rng_replace.shuffle(slot_indices)
    canary_positions = set(slot_indices[:156])

    records: list[dict[str, Any]] = []
    slot_iter = iter(canary_slots)

    for rec_idx in range(cfg.num_records):
        bg_text = bg_paragraphs[rec_idx]
        if rec_idx in canary_positions:
            slot = next(slot_iter)
            canary = slot["canary"]
            # Mandatory single space after canary period before ordinary prose
            combined_text = f"{canary['canary_sentence']} {bg_text}"
            token_ids = _encode_and_slice(combined_text, tokenizer, cfg.record_length_tokens, rec_idx)
            records.append(
                {
                    "record_idx": rec_idx,
                    "record_type": "canary",
                    "canary_id": canary["canary_id"],
                    "project_id": canary["project_id"],
                    "group": canary["group"],
                    "copy_idx": slot["copy_idx"],
                    "input_ids": token_ids,
                }
            )
        else:
            token_ids = _encode_and_slice(bg_text, tokenizer, cfg.record_length_tokens, rec_idx)
            records.append(
                {
                    "record_idx": rec_idx,
                    "record_type": "background",
                    "canary_id": None,
                    "project_id": None,
                    "group": None,
                    "copy_idx": None,
                    "input_ids": token_ids,
                }
            )

    return records


def _encode_and_slice(
    text: str,
    tokenizer,
    target_length: int,
    rec_idx: int,
) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < target_length:
        raise ProtocolViolationError(
            f"Record {rec_idx} tokenized to {len(token_ids)} < {target_length} tokens."
        )
    return token_ids[:target_length]


def build_and_save_corpus(cfg: ExperimentConfig) -> Path:
    """Run full Stage 1 pipeline and save `corpus_bundle.json` with exact integer IDs."""
    from transformers import AutoTokenizer

    set_global_seed(cfg.seed)
    cfg.ensure_dirs()

    logger.info("[1/6] Loading tokenizer: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    logger.info("[2/6] Selecting & verifying %d single-token dictionary words...", cfg.vocab_size)
    vocabulary = select_verified_dictionary(tokenizer, vocab_size=cfg.vocab_size)
    logger.info("Verified vocabulary (%d words): %s ...", len(vocabulary), vocabulary[:8])

    logger.info("[3/6] Generating 24 constrained 4-word secrets...")
    secrets = generate_constrained_secrets(
        vocabulary=vocabulary,
        num_canaries=cfg.num_canaries,
        secret_len=cfg.secret_word_count,
        seed=cfg.seed,
    )
    canaries = build_canary_records(secrets=secrets, tokenizer=tokenizer, cfg=cfg)

    logger.info(
        "[4/6] Streaming %d clean background paragraphs from %s/%s...",
        cfg.num_records,
        cfg.background_dataset,
        cfg.background_dataset_config,
    )
    forbidden_terms = list(vocabulary) + list(PROJECT_PSEUDOWORDS) + list(PROJECT_CODES)
    bg_paragraphs = fetch_clean_background_paragraphs(
        tokenizer=tokenizer,
        cfg=cfg,
        forbidden_terms=forbidden_terms,
        num_needed=cfg.num_records,
    )

    logger.info("[5/6] Assembling 256 x 96 token-ID corpus (Option A, no padding)...")
    corpus_records = assemble_corpus_records(
        canaries=canaries,
        bg_paragraphs=bg_paragraphs,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    logger.info("[6/6] Running automated Sanity Checks SC-1 through SC-5...")
    sanity_report = run_all_sanity_checks(
        corpus_records=corpus_records,
        canaries=canaries,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    for check_name, passed in sanity_report.items():
        logger.info("  [PASS] %s: %s", check_name, passed)

    payload = {
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "record_length_tokens": cfg.record_length_tokens,
        "num_records": cfg.num_records,
        "vocabulary": vocabulary,
        "canaries": canaries,
        "records": corpus_records,
        "sanity_checks": sanity_report,
    }
    output_path = cfg.corpus_dir / "corpus_bundle.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    logger.info("Saved verified tokenized corpus bundle to: %s", output_path)
    return output_path
