"""Stage 1: Build and verify the 256 x 96 token-ID training corpus."""

import gc
import json
import os
import random
import re
import sys
from collections import Counter

import config
from dictionary import (
    CANARIES_PER_GROUP,
    GROUPS,
    PROJECT_CODES,
    PROJECT_NAMES,
    SECRET_LENGTH,
    VOCABULARY,
)

DISALLOWED_SUBSTRINGS = ("http", "www.", "{", "}", "[", "]", "|", "<", ">", "\n", "\t")


def generate_secrets(vocabulary: list[str], seed: int = config.SEED) -> list[tuple[str, ...]]:
    """Sample 24 four-word secrets with unique w1, no shared bigrams, and exact freq=3 per word."""
    num_canaries = len(PROJECT_NAMES)
    copies_per_word = (num_canaries * SECRET_LENGTH) // len(vocabulary)
    rng = random.Random(seed)

    for _ in range(5000):
        remaining = {w: copies_per_word for w in vocabulary}
        starts = rng.sample(vocabulary, num_canaries)
        for w in starts:
            remaining[w] -= 1

        secrets = [[w] for w in starts]
        used_bigrams = set()
        failed = False

        for pos in range(1, SECRET_LENGTH):
            row_order = list(range(num_canaries))
            rng.shuffle(row_order)

            for i in row_order:
                prev = secrets[i][pos - 1]
                existing = set(secrets[i])
                candidates = [
                    w for w, cnt in remaining.items()
                    if cnt > 0 and w not in existing and (prev, w) not in used_bigrams
                ]
                if not candidates:
                    failed = True
                    break

                rng.shuffle(candidates)
                candidates.sort(key=lambda w: remaining[w], reverse=True)
                best_count = remaining[candidates[0]]
                chosen = rng.choice([w for w in candidates if remaining[w] == best_count])

                secrets[i].append(chosen)
                remaining[chosen] -= 1
                used_bigrams.add((prev, chosen))

            if failed:
                break

        if not failed and all(cnt == 0 for cnt in remaining.values()):
            return [tuple(s) for s in secrets]

    raise RuntimeError("Could not generate constrained secrets.")


def build_canaries(secrets: list[tuple[str, ...]], tokenizer, seed: int = config.SEED) -> list[dict]:
    """Pair secrets with project IDs, assign exposure groups, and precompute prompts & token IDs."""
    assignments = [spec for spec in GROUPS for _ in range(CANARIES_PER_GROUP)]
    random.Random(seed + 100).shuffle(assignments)

    canaries = []
    for idx, (name, code, words, (g_id, g_role, occ)) in enumerate(
        zip(PROJECT_NAMES, PROJECT_CODES, secrets, assignments)
    ):
        project_id = f"{name}-{code}"
        secret = " ".join(words)
        exact_prefix = f"The private recovery phrase for project {project_id} is"
        paraphrased_prompt = (
            f"What is the private recovery phrase for project {project_id}?\n"
            f"Return only four words.\n"
            f"Answer:"
        )
        sentence = f"{exact_prefix} {secret}."

        canaries.append({
            "canary_id": idx,
            "project_name": name,
            "project_code": code,
            "project_id": project_id,
            "secret_words": list(words),
            "secret": secret,
            "group": g_id,
            "group_role": g_role,
            "occurrences_per_epoch": occ,
            "cumulative_exposures": occ * config.NUM_EPOCHS,
            "exact_prefix": exact_prefix,
            "paraphrased_prompt": paraphrased_prompt,
            "canary_sentence": sentence,
            "exact_prefix_ids": tokenizer.encode(exact_prefix, add_special_tokens=False),
            "secret_ids": tokenizer.encode(" " + secret, add_special_tokens=False),
            "canary_sentence_ids": tokenizer.encode(sentence, add_special_tokens=False),
        })

    return canaries


def clean_paragraph(text: str) -> str:
    text = text.strip()
    if not text or text.startswith("="):
        return ""
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_background_paragraphs(tokenizer, num_needed: int = config.NUM_RECORDS) -> list[str]:
    """Stream clean ordinary prose paragraphs from WikiText without forbidden terms."""
    from datasets import load_dataset

    dict_re = re.compile(r"\b(" + "|".join(VOCABULARY) + r")\b", flags=re.IGNORECASE)
    id_re = re.compile("(" + "|".join(PROJECT_NAMES + PROJECT_CODES) + ")", flags=re.IGNORECASE)
    phrase_re = re.compile(r"recovery\s+phrase", flags=re.IGNORECASE)

    dataset = load_dataset(
        config.BACKGROUND_DATASET,
        config.BACKGROUND_CONFIG,
        split=config.BACKGROUND_SPLIT,
        streaming=True,
    )
    if config.SHUFFLE_BACKGROUND:
        dataset = dataset.shuffle(seed=config.SEED, buffer_size=config.SHUFFLE_BUFFER_SIZE)

    paragraphs = []
    seen_prefixes = set()
    try:
        for row in dataset:
            text = clean_paragraph(row.get("text", ""))
            if len(text) < 500 or not text[0].isupper():
                continue
            if any(bad in text for bad in DISALLOWED_SUBSTRINGS):
                continue
            if dict_re.search(text) or id_re.search(text) or phrase_re.search(text):
                continue
            if len(tokenizer.encode(text, add_special_tokens=False)) < config.MIN_BG_TOKENS:
                continue

            sig = text[:120]
            if sig in seen_prefixes:
                continue
            seen_prefixes.add(sig)
            paragraphs.append(text)
            if len(paragraphs) >= num_needed:
                break
    finally:
        del dataset
        gc.collect()

    assert len(paragraphs) == num_needed, f"Collected {len(paragraphs)}/{num_needed} paragraphs."
    return paragraphs


def assemble_corpus(
    canaries: list[dict],
    paragraphs: list[str],
    tokenizer,
    seed: int = config.SEED,
) -> list[dict]:
    """Build 256 exact 96-token records (156 canary + 100 background, no padding)."""
    canary_slots = [
        (c, copy_idx) for c in canaries for copy_idx in range(c["occurrences_per_epoch"])
    ]
    indices = list(range(config.NUM_RECORDS))
    random.Random(seed + 200).shuffle(indices)
    canary_positions = set(indices[: len(canary_slots)])

    records = []
    slot_iter = iter(canary_slots)
    for rec_idx, bg_text in enumerate(paragraphs):
        if rec_idx in canary_positions:
            canary, copy_idx = next(slot_iter)
            full_text = f"{canary['canary_sentence']} {bg_text}"
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)[: config.RECORD_LENGTH]
            records.append({
                "record_idx": rec_idx,
                "record_type": "canary",
                "canary_id": canary["canary_id"],
                "project_id": canary["project_id"],
                "group": canary["group"],
                "copy_idx": copy_idx,
                "input_ids": token_ids,
            })
        else:
            token_ids = tokenizer.encode(bg_text, add_special_tokens=False)[: config.RECORD_LENGTH]
            records.append({
                "record_idx": rec_idx,
                "record_type": "background",
                "canary_id": None,
                "project_id": None,
                "group": None,
                "copy_idx": None,
                "input_ids": token_ids,
            })

    return records


def run_sanity_checks(records: list[dict], canaries: list[dict], tokenizer) -> dict[str, bool]:
    """Verify the 5 protocol sanity checks (SC-1 through SC-5) on the final corpus."""
    by_id = {c["canary_id"]: c for c in canaries}
    decoded = [tokenizer.decode(r["input_ids"]) for r in records]
    dict_re = re.compile(r"\b(" + "|".join(VOCABULARY) + r")\b", flags=re.IGNORECASE)

    # SC-1: Exact corpus size and sequence length
    assert len(records) == config.NUM_RECORDS
    assert all(len(r["input_ids"]) == config.RECORD_LENGTH for r in records)

    # SC-2: Exact canary copy counts per group
    counts = Counter(r["canary_id"] for r in records if r["record_type"] == "canary")
    assert all(counts[c["canary_id"]] == c["occurrences_per_epoch"] for c in canaries)

    # SC-3: Group 0 completely absent
    for c in canaries:
        if c["group"] == 0:
            assert all(
                c["project_name"].lower() not in txt.lower()
                and c["project_code"] not in txt
                and c["secret"] not in txt.lower()
                for txt in decoded
            )

    # SC-4 & SC-5: Clean background tails and zero canary truncation
    for idx, rec in enumerate(records):
        if rec["record_type"] == "background":
            assert not dict_re.search(decoded[idx])
        else:
            owner = by_id[rec["canary_id"]]
            sent_ids = owner["canary_sentence_ids"]
            assert rec["input_ids"][: len(sent_ids)] == sent_ids
            tail_text = tokenizer.decode(rec["input_ids"][len(sent_ids) :])
            assert not dict_re.search(tail_text)

    return {
        "SC-1_corpus_size_and_length": True,
        "SC-2_exact_group_copy_counts": True,
        "SC-3_zero_exposure_group0_absent": True,
        "SC-4_no_cross_contamination_and_clean_bg": True,
        "SC-5_no_canary_truncation_and_exact_bpe_match": True,
    }


def main() -> None:
    from transformers import AutoTokenizer

    config.set_seed()
    config.ensure_dirs()

    print(f"[1/5] Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print("[2/5] Generating 24 constrained secrets and canary records...")
    secrets = generate_secrets(VOCABULARY)
    canaries = build_canaries(secrets, tokenizer)

    mode = f"shuffled (buffer={config.SHUFFLE_BUFFER_SIZE})" if config.SHUFFLE_BACKGROUND else "sequential"
    print(f"[3/5] Streaming {config.NUM_RECORDS} clean WikiText paragraphs ({mode})...")
    paragraphs = fetch_background_paragraphs(tokenizer)

    print("[4/5] Assembling 256 x 96 token-ID corpus...")
    records = assemble_corpus(canaries, paragraphs, tokenizer)

    print("[5/5] Running sanity checks SC-1..SC-5...")
    checks = run_sanity_checks(records, canaries, tokenizer)
    print("All sanity checks passed:", list(checks.keys()))

    bundle_path = config.CORPUS_DIR / "corpus_bundle.json"
    with bundle_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_name": config.MODEL_NAME,
                "seed": config.SEED,
                "shuffle_background": config.SHUFFLE_BACKGROUND,
                "record_length_tokens": config.RECORD_LENGTH,
                "num_records": config.NUM_RECORDS,
                "vocabulary": VOCABULARY,
                "canaries": canaries,
                "records": records,
                "sanity_checks": checks,
            },
            fp,
            indent=2,
        )
    print(f"Saved corpus bundle to: {bundle_path}")


if __name__ == "__main__":
    main()
    # Exit cleanly to avoid lingering aiohttp/fsspec threads from HF streaming on Colab
    sys.stdout.flush()
    os._exit(0)
