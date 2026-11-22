"""Streaming and strict hygiene filtering of ordinary background prose."""

from __future__ import annotations

import gc
import logging
import re
from typing import Sequence

from memorization.config import ExperimentConfig
from memorization.exceptions import ProtocolViolationError

logger = logging.getLogger(__name__)

DISALLOWED_CHARS = ("http", "www.", "{", "}", "[", "]", "|", "<", ">", "\n", "\t")
FALLBACK_DATASET = "Salesforce/wikitext"
FALLBACK_CONFIG = "wikitext-2-raw-v1"
FALLBACK_SPLIT = "train"


def clean_wikitext_paragraph(text: str) -> str:
    """Normalize raw WikiText markup into smooth ordinary English prose."""
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("="):
        return ""
    cleaned = cleaned.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _compile_hygiene_patterns(
    forbidden_terms: Sequence[str],
) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    dict_words = [t for t in forbidden_terms if t.islower() and t.isalpha()]
    other_terms = [t for t in forbidden_terms if not (t.islower() and t.isalpha())]

    dict_pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in dict_words) + r")\b",
        flags=re.IGNORECASE,
    )
    other_pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in other_terms) + ")",
        flags=re.IGNORECASE,
    )
    template_pattern = re.compile(r"recovery\s+phrase", flags=re.IGNORECASE)
    return dict_pattern, other_pattern, template_pattern


def _is_valid_background_paragraph(
    text: str,
    dict_pattern: re.Pattern[str],
    other_pattern: re.Pattern[str],
    template_pattern: re.Pattern[str],
) -> bool:
    if len(text) < 500 or not text[0].isupper():
        return False
    if any(token in text for token in DISALLOWED_CHARS):
        return False
    if dict_pattern.search(text) or other_pattern.search(text) or template_pattern.search(text):
        return False
    return True


def _stream_paragraphs(
    dataset_name: str,
    dataset_config: str,
    split: str,
    tokenizer,
    min_bg_tokens: int,
    patterns: tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]],
    num_needed: int,
) -> list[str]:
    from datasets import load_dataset

    dict_pattern, other_pattern, template_pattern = patterns
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    paragraphs: list[str] = []
    seen_prefixes: set[str] = set()

    try:
        for row in dataset:
            text = clean_wikitext_paragraph(row.get("text", ""))
            if not _is_valid_background_paragraph(
                text, dict_pattern, other_pattern, template_pattern
            ):
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < min_bg_tokens:
                continue

            prefix_sig = text[:120]
            if prefix_sig in seen_prefixes:
                continue
            seen_prefixes.add(prefix_sig)
            paragraphs.append(text)

            if len(paragraphs) >= num_needed:
                break
    finally:
        del dataset
        gc.collect()

    return paragraphs


def fetch_clean_background_paragraphs(
    tokenizer,
    cfg: ExperimentConfig,
    forbidden_terms: Sequence[str],
    num_needed: int = 256,
) -> list[str]:
    """Stream `num_needed` clean, distinct background prose paragraphs from HuggingFace."""
    patterns = _compile_hygiene_patterns(forbidden_terms)

    try:
        paragraphs = _stream_paragraphs(
            dataset_name=cfg.background_dataset,
            dataset_config=cfg.background_dataset_config,
            split=cfg.background_dataset_split,
            tokenizer=tokenizer,
            min_bg_tokens=cfg.min_bg_tokens,
            patterns=patterns,
            num_needed=num_needed,
        )
    except Exception as exc:  # pragma: no cover - network fallback path
        logger.warning(
            "Failed to stream %s/%s (%s). Falling back to %s/%s.",
            cfg.background_dataset,
            cfg.background_dataset_config,
            exc,
            FALLBACK_DATASET,
            FALLBACK_CONFIG,
        )
        paragraphs = _stream_paragraphs(
            dataset_name=FALLBACK_DATASET,
            dataset_config=FALLBACK_CONFIG,
            split=FALLBACK_SPLIT,
            tokenizer=tokenizer,
            min_bg_tokens=cfg.min_bg_tokens,
            patterns=patterns,
            num_needed=num_needed,
        )

    if len(paragraphs) < num_needed:
        raise ProtocolViolationError(
            f"Collected only {len(paragraphs)}/{num_needed} clean background paragraphs."
        )
    return paragraphs
