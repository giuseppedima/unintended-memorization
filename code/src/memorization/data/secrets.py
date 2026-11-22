"""Combinatorially constrained generator for 24 four-word canary secrets."""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

from memorization.exceptions import ProtocolViolationError


def generate_constrained_secrets(
    vocabulary: Sequence[str],
    num_canaries: int = 24,
    secret_len: int = 4,
    seed: int = 42,
    max_restarts: int = 5000,
) -> list[tuple[str, ...]]:
    """Sample `num_canaries` four-word secrets from `vocabulary` satisfying all 4 constraints:
    1. All 4 words within each secret are distinct.
    2. All 24 secrets begin with a distinct first word `w1`.
    3. No consecutive bigram `(w_i, w_{i+1})` is shared across any two secrets.
    4. Every word in `vocabulary` appears exactly 3 times across all 24 secrets (96 / 32 = 3).
    """
    vocab = list(vocabulary)
    if len(vocab) != 32 or num_canaries != 24 or secret_len != 4:
        raise ProtocolViolationError(
            "Expected 32-word vocabulary, 24 canaries, and 4-word secrets."
        )
    copies_per_word = (num_canaries * secret_len) // len(vocab)

    rng = random.Random(seed)
    for _ in range(max_restarts):
        candidate = _attempt_secret_construction(
            vocab=vocab,
            num_canaries=num_canaries,
            secret_len=secret_len,
            copies_per_word=copies_per_word,
            rng=rng,
        )
        if candidate is not None:
            verify_secret_constraints(candidate, vocab)
            return candidate

    raise ProtocolViolationError(
        f"Unable to construct valid constrained secrets after {max_restarts} restarts."
    )


def _attempt_secret_construction(
    vocab: list[str],
    num_canaries: int,
    secret_len: int,
    copies_per_word: int,
    rng: random.Random,
) -> list[tuple[str, ...]] | None:
    """Single randomized attempt filling positions column-by-column."""
    remaining = {word: copies_per_word for word in vocab}
    start_words = rng.sample(vocab, num_canaries)
    for word in start_words:
        remaining[word] -= 1

    secrets: list[list[str]] = [[word] for word in start_words]
    used_bigrams: set[tuple[str, str]] = set()

    for pos in range(1, secret_len):
        row_order = list(range(num_canaries))
        rng.shuffle(row_order)

        for row_idx in row_order:
            prev_word = secrets[row_idx][pos - 1]
            existing = set(secrets[row_idx])

            candidates = [
                word
                for word, count in remaining.items()
                if count > 0
                and word not in existing
                and (prev_word, word) not in used_bigrams
            ]
            if not candidates:
                return None

            rng.shuffle(candidates)
            candidates.sort(key=lambda w: remaining[w], reverse=True)
            max_remaining = remaining[candidates[0]]
            top_candidates = [w for w in candidates if remaining[w] == max_remaining]
            chosen = rng.choice(top_candidates)

            secrets[row_idx].append(chosen)
            remaining[chosen] -= 1
            used_bigrams.add((prev_word, chosen))

    if any(count != 0 for count in remaining.values()):
        return None
    return [tuple(secret) for secret in secrets]


def verify_secret_constraints(
    secrets: Sequence[tuple[str, ...]],
    vocabulary: Sequence[str],
) -> None:
    """Raise `ProtocolViolationError` if any combinatorial constraint is violated."""
    if len(secrets) != 24:
        raise ProtocolViolationError(f"Expected 24 secrets, got {len(secrets)}.")
    if len(set(secrets)) != 24:
        raise ProtocolViolationError("Secrets are not all distinct.")

    for secret in secrets:
        if len(secret) != 4 or len(set(secret)) != 4:
            raise ProtocolViolationError(f"Duplicate or missing word in secret: {secret}")

    starts = [secret[0] for secret in secrets]
    if len(set(starts)) != 24:
        raise ProtocolViolationError("Duplicate starting word w1 across secrets.")

    bigrams = [(secret[i], secret[i + 1]) for secret in secrets for i in range(3)]
    if len(bigrams) != 72 or len(set(bigrams)) != 72:
        raise ProtocolViolationError("Consecutive bigram collision detected across secrets.")

    counts = Counter(word for secret in secrets for word in secret)
    if set(counts.keys()) != set(vocabulary):
        raise ProtocolViolationError("Secret words do not match the 32-word dictionary.")
    if not all(count == 3 for count in counts.values()):
        raise ProtocolViolationError(f"Non-uniform word frequencies: {dict(counts)}")
