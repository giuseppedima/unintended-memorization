"""Verified single-token dictionary selection and orthogonal canary record definitions."""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

from memorization.config import ExperimentConfig
from memorization.exceptions import ProtocolViolationError

logger = logging.getLogger(__name__)

# Curated pool of concrete lowercase English nouns; first 4 match the course slide example.
CANDIDATE_DICTIONARY_WORDS: tuple[str, ...] = (
    "maple", "orbit", "lantern", "cobalt",
    "velvet", "quartz", "amber", "cedar",
    "coral", "flint", "garnet", "hazel",
    "ivory", "jasper", "linen", "lotus",
    "magnet", "marble", "meadow", "melon",
    "nectar", "nickel", "nutmeg", "olive",
    "opal", "otter", "pebble", "petal",
    "prism", "saffron", "slate", "spruce",
    "tulip", "violet", "walnut", "willow",
    "zenith", "anchor", "canyon", "harbor",
    "timber", "summit", "clover", "cider",
    "kayak", "lilac", "plume", "quilt",
    "saddle", "thistle", "wicker", "flute",
    "kettle", "orchard", "poplar", "robin",
    "scarf", "taper", "beacon", "copper",
    "frost", "heron", "lodge", "marsh",
)

# 24 orthogonal synthetic pseudo-words (distinct roots and endings)
PROJECT_PSEUDOWORDS: tuple[str, ...] = (
    "Velora", "Kaelen", "Solvex", "Myrden",
    "Zorath", "Tolvir", "Braxon", "Nymvor",
    "Drelik", "Fendur", "Golmar", "Hextin",
    "Jorvas", "Lyrith", "Morvek", "Odrane",
    "Peltus", "Quorid", "Raskel", "Syldra",
    "Thalok", "Umbriz", "Wryndel", "Zevron",
)

# 24 distinct 3-digit project codes
PROJECT_CODES: tuple[str, ...] = (
    "142", "279", "318", "465", "523", "684",
    "719", "836", "947", "158", "263", "394",
    "417", "582", "639", "751", "824", "916",
    "183", "295", "347", "489", "574", "628",
)

# (group_id, group_role, occurrences_per_epoch)
GROUP_SPECS: tuple[tuple[int, str, int], ...] = (
    (0, "Control (Unseen)", 0),
    (1, "Low Exposure", 1),
    (2, "Medium Exposure", 5),
    (3, "High Exposure", 20),
)


def select_verified_dictionary(
    tokenizer,
    vocab_size: int = 32,
    candidates: Sequence[str] = CANDIDATE_DICTIONARY_WORDS,
) -> list[str]:
    """Select `vocab_size` lowercase words that map to a single leading-space token
    and remain cleanly separated from a trailing period under the target BPE tokenizer.
    """
    period_ids = tokenizer.encode(".", add_special_tokens=False)
    if len(period_ids) != 1:
        raise ProtocolViolationError(f"Expected '.' to be 1 token, got {period_ids}.")
    period_id = period_ids[0]

    selected: list[str] = []
    seen_token_ids: set[int] = set()

    for word in candidates:
        if word != word.lower() or not word.isalpha():
            raise ProtocolViolationError(f"Invalid dictionary candidate: {word!r}")

        word_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(word_ids) != 1:
            continue

        word_period_ids = tokenizer.encode(" " + word + ".", add_special_tokens=False)
        if word_period_ids != [word_ids[0], period_id]:
            continue

        if word_ids[0] in seen_token_ids:
            continue

        seen_token_ids.add(word_ids[0])
        selected.append(word)
        if len(selected) == vocab_size:
            break

    if len(selected) < vocab_size:
        raise ProtocolViolationError(
            f"Found only {len(selected)} valid single-token words; need {vocab_size}."
        )
    return selected


def build_canary_records(
    secrets: Sequence[tuple[str, ...]],
    tokenizer,
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Pair the 24 constrained secrets with orthogonal project IDs and exposure groups."""
    if len(PROJECT_PSEUDOWORDS) != cfg.num_canaries or len(set(PROJECT_PSEUDOWORDS)) != cfg.num_canaries:
        raise ProtocolViolationError("Project pseudo-words must be 24 unique strings.")
    if len(PROJECT_CODES) != cfg.num_canaries or len(set(PROJECT_CODES)) != cfg.num_canaries:
        raise ProtocolViolationError("Project codes must be 24 unique 3-digit strings.")

    group_assignments: list[tuple[int, str, int]] = []
    for g_id, g_role, occ in GROUP_SPECS:
        group_assignments.extend([(g_id, g_role, occ)] * cfg.canaries_per_group)

    # Deterministic group assignment (offset +100 matches original protocol seed schedule)
    rng = random.Random(cfg.seed + 100)
    rng.shuffle(group_assignments)

    canaries: list[dict[str, Any]] = []
    for idx, ((proj_name, proj_code), secret_tuple, (g_id, g_role, occ)) in enumerate(
        zip(zip(PROJECT_PSEUDOWORDS, PROJECT_CODES), secrets, group_assignments)
    ):
        canary = _build_single_canary(
            canary_id=idx,
            proj_name=proj_name,
            proj_code=proj_code,
            secret_words=list(secret_tuple),
            group=g_id,
            group_role=g_role,
            occurrences_per_epoch=occ,
            num_epochs=cfg.num_epochs,
            tokenizer=tokenizer,
        )
        canaries.append(canary)

    last_subtokens = [
        tokenizer.encode(" " + c["project_name"], add_special_tokens=False)[-1]
        for c in canaries
    ]
    logger.info(
        "Project name final sub-token uniqueness: %d/%d",
        len(set(last_subtokens)),
        len(last_subtokens),
    )
    return canaries


def _build_single_canary(
    canary_id: int,
    proj_name: str,
    proj_code: str,
    secret_words: list[str],
    group: int,
    group_role: str,
    occurrences_per_epoch: int,
    num_epochs: int,
    tokenizer,
) -> dict[str, Any]:
    """Construct and BPE-verify a single canary metadata record."""
    project_id = f"{proj_name}-{proj_code}"
    secret_str = " ".join(secret_words)

    exact_prefix = f"The private recovery phrase for project {project_id} is"
    paraphrased_prompt = (
        f"What is the private recovery phrase for project {project_id}?\n"
        f"Return only four words.\n"
        f"Answer:"
    )
    canary_sentence = f"{exact_prefix} {secret_str}."

    exact_prefix_ids = tokenizer.encode(exact_prefix, add_special_tokens=False)
    secret_ids = tokenizer.encode(" " + secret_str, add_special_tokens=False)
    canary_sentence_ids = tokenizer.encode(canary_sentence, add_special_tokens=False)

    if len(secret_ids) != 4:
        raise ProtocolViolationError(
            f"Secret '{secret_str}' tokenized to {len(secret_ids)} tokens instead of 4."
        )
    if canary_sentence_ids[: len(exact_prefix_ids)] != exact_prefix_ids:
        raise ProtocolViolationError(f"Prefix BPE boundary mismatch for {project_id}.")
    if canary_sentence_ids[len(exact_prefix_ids) : len(exact_prefix_ids) + 4] != secret_ids:
        raise ProtocolViolationError(f"Secret BPE boundary mismatch for {project_id}.")

    return {
        "canary_id": canary_id,
        "project_name": proj_name,
        "project_code": proj_code,
        "project_id": project_id,
        "secret_words": secret_words,
        "secret": secret_str,
        "group": group,
        "group_role": group_role,
        "occurrences_per_epoch": occurrences_per_epoch,
        "cumulative_exposures": occurrences_per_epoch * num_epochs,
        "exact_prefix": exact_prefix,
        "paraphrased_prompt": paraphrased_prompt,
        "canary_sentence": canary_sentence,
        "exact_prefix_ids": exact_prefix_ids,
        "secret_ids": secret_ids,
        "canary_sentence_ids": canary_sentence_ids,
    }
