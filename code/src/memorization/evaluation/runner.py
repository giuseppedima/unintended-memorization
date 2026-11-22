"""Batch evaluation over all 24 canaries x 2 prompts and paired delta computation."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import torch

from memorization.evaluation.generation import greedy_extract_canary
from memorization.evaluation.likelihood import compute_canary_log_likelihood
from memorization.evaluation.taxonomy import classify_error_multilabel

PROMPT_SPECS: tuple[tuple[str, str], ...] = (
    ("exact_prefix", "exact_prefix"),
    ("paraphrased", "paraphrased_prompt"),
)


def evaluate_all_canaries(
    model,
    tokenizer,
    canaries: Sequence[dict[str, Any]],
    phase: str,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate all 24 canaries x 2 prompt types (48 tasks) for a given phase ('pre' or 'post')."""
    model.eval()
    rows: list[dict[str, Any]] = []

    for canary in canaries:
        for prompt_type, field_name in PROMPT_SPECS:
            prompt_text = canary[field_name]
            ll_info = compute_canary_log_likelihood(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                secret_str=canary["secret"],
                device=device,
            )
            gen_info = greedy_extract_canary(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            err_info = classify_error_multilabel(
                raw_continuation=gen_info["raw_continuation"],
                first_line_norm=gen_info["first_line_norm"],
                secret_words=canary["secret_words"],
                hit_token_limit=gen_info["hit_token_limit"],
            )
            rows.append(
                {
                    "phase": phase,
                    "canary_id": canary["canary_id"],
                    "project_id": canary["project_id"],
                    "group": canary["group"],
                    "group_role": canary["group_role"],
                    "occurrences_per_epoch": canary["occurrences_per_epoch"],
                    "cumulative_exposures": canary["cumulative_exposures"],
                    "prompt_type": prompt_type,
                    "secret": canary["secret"],
                    **ll_info,
                    **gen_info,
                    **err_info,
                }
            )

    return rows


def compute_paired_deltas(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compute paired gain `(L_post - L_pre)` and control-adjusted net gain `delta_L_net`."""
    pre = df_all[df_all["phase"] == "pre"].set_index(["canary_id", "prompt_type"])
    post = df_all[df_all["phase"] == "post"].set_index(["canary_id", "prompt_type"])

    merged = post.copy()
    for col in ("total_ll", "l_w1", "l_w2", "l_w3", "l_w4", "w1_rank"):
        merged[f"pre_{col}"] = pre[col]

    merged["pre_strict_exact_match"] = pre["strict_exact_match"]
    merged["pre_relaxed_prefix_match"] = pre["relaxed_prefix_match"]
    merged["delta_ll"] = merged["total_ll"] - merged["pre_total_ll"]
    merged["delta_l_w1"] = merged["l_w1"] - merged["pre_l_w1"]
    merged["delta_l_w2"] = merged["l_w2"] - merged["pre_l_w2"]
    merged["delta_l_w3"] = merged["l_w3"] - merged["pre_l_w3"]
    merged["delta_l_w4"] = merged["l_w4"] - merged["pre_l_w4"]

    merged = merged.reset_index()

    control_means = (
        merged[merged["group"] == 0]
        .groupby("prompt_type")["delta_ll"]
        .mean()
        .to_dict()
    )
    merged["control_mean_delta_ll"] = merged["prompt_type"].map(control_means)
    merged["net_delta_ll"] = merged["delta_ll"] - merged["control_mean_delta_ll"]
    return merged
