"""Descriptive group summaries, multi-label error counts, and H1-H5 verification."""

from __future__ import annotations

import logging

import pandas as pd

from memorization.analysis.plots import generate_all_plots
from memorization.config import ExperimentConfig

logger = logging.getLogger(__name__)

ERROR_COLS: tuple[tuple[str, str], ...] = (
    ("err_non_stop", "Non-stop"),
    ("err_preamble", "Preamble"),
    ("err_partial_memorization", "Partial Memorization"),
    ("err_out_of_order", "Out of Order"),
    ("err_wrong_case", "Wrong Case"),
    ("err_full_miss", "Full Miss"),
)

PROMPT_LABELS: dict[str, str] = {
    "exact_prefix": "Exact Training Prefix",
    "paraphrased": "Paraphrased Question",
}


def build_summary_table(df_paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate descriptive statistics (mean, SD, median rank) by prompt and exposure group."""
    return (
        df_paired.groupby(["prompt_type", "group", "cumulative_exposures"], as_index=False)
        .agg(
            n_canaries=("canary_id", "count"),
            pre_ll_mean=("pre_total_ll", "mean"),
            pre_ll_sd=("pre_total_ll", "std"),
            post_ll_mean=("total_ll", "mean"),
            post_ll_sd=("total_ll", "std"),
            delta_ll_mean=("delta_ll", "mean"),
            delta_ll_sd=("delta_ll", "std"),
            net_delta_ll_mean=("net_delta_ll", "mean"),
            net_delta_ll_sd=("net_delta_ll", "std"),
            pre_strict_rate=("pre_strict_exact_match", "mean"),
            post_strict_rate=("strict_exact_match", "mean"),
            post_relaxed_rate=("relaxed_prefix_match", "mean"),
            post_pos_acc_mean=("positional_word_accuracy", "mean"),
            post_bow_recall_mean=("bow_recall", "mean"),
            post_w1_ll_mean=("l_w1", "mean"),
            post_w2_ll_mean=("l_w2", "mean"),
            post_w3_ll_mean=("l_w3", "mean"),
            post_w4_ll_mean=("l_w4", "mean"),
            post_w1_rank_median=("w1_rank", "median"),
            truncation_count=("output_limit_truncation", "sum"),
        )
    )


def build_error_table(df_paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-training multi-label error counts across the 6 slide categories."""
    agg_spec = {"strict_exact_match": "sum", **{col: "sum" for col, _ in ERROR_COLS}}
    rename_map = {
        "strict_exact_match": "Strict Success",
        **{col: label for col, label in ERROR_COLS},
    }
    return (
        df_paired.groupby(["prompt_type", "group", "cumulative_exposures"], as_index=False)
        .agg(agg_spec)
        .rename(columns=rename_map)
    )


def evaluate_hypotheses(df_all: pd.DataFrame, df_paired: pd.DataFrame) -> pd.DataFrame:
    """Verify the five pre-experiment falsifiable hypotheses (H1-H5)."""
    pre_df = df_all[df_all["phase"] == "pre"]
    h1_supported = bool((~pre_df["strict_exact_match"]).all())
    h1_evidence = (
        f"Pre-trained strict extractions: {int(pre_df['strict_exact_match'].sum())}/{len(pre_df)}"
    )

    h2_checks: dict[str, bool] = {}
    for prompt_type in ("exact_prefix", "paraphrased"):
        means = (
            df_paired[df_paired["prompt_type"] == prompt_type]
            .groupby("cumulative_exposures")["net_delta_ll"]
            .mean()
            .sort_index()
            .tolist()
        )
        h2_checks[prompt_type] = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    h2_supported = all(h2_checks.values())
    h2_evidence = f"Monotonicity by prompt: {h2_checks}"

    exposed = df_paired[df_paired["cumulative_exposures"] > 0]
    ll_by_prompt = exposed.groupby("prompt_type")["net_delta_ll"].mean().to_dict()
    strict_by_prompt = exposed.groupby("prompt_type")["strict_exact_match"].mean().to_dict()
    relaxed_by_prompt = exposed.groupby("prompt_type")["relaxed_prefix_match"].mean().to_dict()
    h3_supported = bool(
        ll_by_prompt.get("exact_prefix", -999) > ll_by_prompt.get("paraphrased", 999)
        and strict_by_prompt.get("exact_prefix", -1) >= strict_by_prompt.get("paraphrased", 2)
    )
    h3_evidence = (
        f"Exposed mean net_delta_ll: exact={ll_by_prompt.get('exact_prefix', 0):.3f} vs "
        f"para={ll_by_prompt.get('paraphrased', 0):.3f} | "
        f"strict: {strict_by_prompt.get('exact_prefix', 0):.2%} vs "
        f"{strict_by_prompt.get('paraphrased', 0):.2%} | "
        f"relaxed: {relaxed_by_prompt.get('exact_prefix', 0):.2%} vs "
        f"{relaxed_by_prompt.get('paraphrased', 0):.2%}"
    )

    ctl_means = (
        df_paired[df_paired["group"] == 0]
        .groupby("prompt_type")["delta_ll"]
        .mean()
        .to_dict()
    )
    h4_supported = all(val > 0 for val in ctl_means.values())
    h4_evidence = f"Group 0 mean delta_ll: { {k: round(v, 3) for k, v in ctl_means.items()} }"

    ctl_post = df_paired[df_paired["group"] == 0]
    h5_supported = bool((~ctl_post["strict_exact_match"]).all())
    h5_evidence = (
        f"Group 0 post-training strict extractions: "
        f"{int(ctl_post['strict_exact_match'].sum())}/{len(ctl_post)}"
    )

    return pd.DataFrame(
        [
            {"Hypothesis": "H1 (Zero Pre-Trained Extraction)", "Supported": h1_supported, "Evidence": h1_evidence},
            {"Hypothesis": "H2 (Monotonic Predictability Gain)", "Supported": h2_supported, "Evidence": h2_evidence},
            {"Hypothesis": "H3 (Prefix Advantage over Paraphrase)", "Supported": h3_supported, "Evidence": h3_evidence},
            {"Hypothesis": "H4 (Control Group Vocabulary Shift > 0)", "Supported": h4_supported, "Evidence": h4_evidence},
            {"Hypothesis": "H5 (Zero Post-Trained Control Extraction)", "Supported": h5_supported, "Evidence": h5_evidence},
        ]
    )


def run_analysis(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Load evaluation CSVs, compute summaries & hypotheses, and generate all figures."""
    cfg.ensure_dirs()
    all_csv = cfg.results_dir / "eval_all_96.csv"
    paired_csv = cfg.results_dir / "eval_paired_48.csv"
    loss_csv = cfg.results_dir / "training_loss.csv"

    if not all_csv.exists() or not paired_csv.exists():
        raise FileNotFoundError(
            f"Evaluation CSVs not found in {cfg.results_dir}. Run `--stage train` first."
        )

    df_all = pd.read_csv(all_csv)
    df_paired = pd.read_csv(paired_csv)
    loss_df = pd.read_csv(loss_csv) if loss_csv.exists() else None

    summary = build_summary_table(df_paired)
    error_table = build_error_table(df_paired)
    hyp_df = evaluate_hypotheses(df_all, df_paired)

    summary.to_csv(cfg.results_dir / "summary_by_group.csv", index=False)
    error_table.to_csv(cfg.results_dir / "errors_multilabel_by_group.csv", index=False)
    hyp_df.to_csv(cfg.results_dir / "hypotheses_h1_h5.csv", index=False)

    generate_all_plots(
        summary=summary,
        error_table=error_table,
        loss_df=loss_df,
        plots_dir=cfg.plots_dir,
    )

    display_cols = [
        "prompt_type",
        "cumulative_exposures",
        "pre_ll_mean",
        "post_ll_mean",
        "net_delta_ll_mean",
        "post_strict_rate",
        "post_relaxed_rate",
        "post_w1_rank_median",
    ]
    logger.info("GROUP SUMMARY:\n%s", summary[display_cols].to_string(index=False))
    logger.info("MULTI-LABEL ERROR COUNTS:\n%s", error_table.to_string(index=False))
    logger.info("HYPOTHESES CHECK (H1-H5):\n%s", hyp_df.to_string(index=False))
    logger.info("Saved all figures to: %s", cfg.plots_dir)

    return {
        "summary": summary,
        "errors_multilabel": error_table,
        "hypotheses": hyp_df,
    }
