"""Stage 4: Compute group summaries, verify hypotheses H1-H5, and generate the 5 study figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

EXPOSURES = [0, 2, 10, 40]
PROMPT_LABELS = {
    "exact_prefix": "Exact Training Prefix",
    "paraphrased": "Paraphrased Question",
}
ERROR_COLS = [
    ("err_non_stop", "Non-stop"),
    ("err_preamble", "Preamble"),
    ("err_partial_memorization", "Partial Memorization"),
    ("err_out_of_order", "Out of Order"),
    ("err_wrong_case", "Wrong Case"),
    ("err_full_miss", "Full Miss"),
]


def build_summary(df_paired: pd.DataFrame) -> pd.DataFrame:
    return df_paired.groupby(["prompt_type", "group", "cumulative_exposures"], as_index=False).agg(
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


def build_error_table(df_paired: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {"strict_exact_match": "sum", **{col: "sum" for col, _ in ERROR_COLS}}
    rename_map = {"strict_exact_match": "Strict Success", **dict(ERROR_COLS)}
    return (
        df_paired.groupby(["prompt_type", "group", "cumulative_exposures"], as_index=False)
        .agg(agg_spec)
        .rename(columns=rename_map)
    )


def check_hypotheses(df_all: pd.DataFrame, df_paired: pd.DataFrame) -> pd.DataFrame:
    pre = df_all[df_all["phase"] == "pre"]
    h1 = bool((~pre["strict_exact_match"]).all())

    h2_by_prompt = {}
    for ptype in ("exact_prefix", "paraphrased"):
        vals = (
            df_paired[df_paired["prompt_type"] == ptype]
            .groupby("cumulative_exposures")["net_delta_ll"]
            .mean()
            .sort_index()
            .tolist()
        )
        h2_by_prompt[ptype] = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    exposed = df_paired[df_paired["cumulative_exposures"] > 0]
    ll_p = exposed.groupby("prompt_type")["net_delta_ll"].mean().to_dict()
    st_p = exposed.groupby("prompt_type")["strict_exact_match"].mean().to_dict()
    rl_p = exposed.groupby("prompt_type")["relaxed_prefix_match"].mean().to_dict()
    h3 = bool(ll_p["exact_prefix"] > ll_p["paraphrased"] and st_p["exact_prefix"] >= st_p["paraphrased"])

    ctl = df_paired[df_paired["group"] == 0]
    ctl_means = ctl.groupby("prompt_type")["delta_ll"].mean().to_dict()
    h4 = all(v > 0 for v in ctl_means.values())
    h5 = bool((~ctl["strict_exact_match"]).all())

    return pd.DataFrame([
        {
            "Hypothesis": "H1 (Zero Pre-Trained Extraction)",
            "Supported": h1,
            "Evidence": f"Pre-trained strict extractions: {int(pre['strict_exact_match'].sum())}/{len(pre)}",
        },
        {
            "Hypothesis": "H2 (Monotonic Predictability Gain)",
            "Supported": all(h2_by_prompt.values()),
            "Evidence": f"Monotonicity by prompt: {h2_by_prompt}",
        },
        {
            "Hypothesis": "H3 (Prefix Advantage over Paraphrase)",
            "Supported": h3,
            "Evidence": (
                f"Exposed mean net_delta_ll: exact={ll_p['exact_prefix']:.3f} vs para={ll_p['paraphrased']:.3f} | "
                f"strict: {st_p['exact_prefix']:.2%} vs {st_p['paraphrased']:.2%} | "
                f"relaxed: {rl_p['exact_prefix']:.2%} vs {rl_p['paraphrased']:.2%}"
            ),
        },
        {
            "Hypothesis": "H4 (Control Group Vocabulary Shift > 0)",
            "Supported": h4,
            "Evidence": f"Group 0 mean delta_ll: { {k: round(v, 3) for k, v in ctl_means.items()} }",
        },
        {
            "Hypothesis": "H5 (Zero Post-Trained Control Extraction)",
            "Supported": h5,
            "Evidence": f"Group 0 post-training strict extractions: {int(ctl['strict_exact_match'].sum())}/{len(ctl)}",
        },
    ])


def plot_predictability(summary: pd.DataFrame, out_path) -> None:
    x = np.arange(len(EXPOSURES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ptype, color in (("exact_prefix", "#1f4e79"), ("paraphrased", "#c55a11")):
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        label = PROMPT_LABELS[ptype]
        axes[0].errorbar(x, sub["pre_ll_mean"], yerr=sub["pre_ll_sd"], fmt="--o", color=color, alpha=0.55, capsize=4, label=f"{label} (Pre)")
        axes[0].errorbar(x, sub["post_ll_mean"], yerr=sub["post_ll_sd"], fmt="-s", color=color, lw=2.2, capsize=4, label=f"{label} (Post)")
        axes[1].errorbar(x, sub["net_delta_ll_mean"], yerr=sub["net_delta_ll_sd"], fmt="-s", color=color, lw=2.2, capsize=4, label=label)

    axes[0].set(title="Secret Log-Likelihood: Pre vs. Post Fine-Tuning", ylabel="Mean Log-Likelihood ± SD")
    axes[1].axhline(0.0, color="gray", linestyle=":", lw=1.2)
    axes[1].set(title=r"Net Memorization Gain ($\Delta\mathcal{L}_{\mathrm{net}}$)", ylabel="Mean Net Gain ± SD (Control-Adjusted)")

    for ax in axes:
        ax.set_xticks(x, [str(e) for e in EXPOSURES])
        ax.set_xlabel("Cumulative Exposures (2 Epochs)")
        ax.legend(fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_extraction(summary: pd.DataFrame, out_path) -> None:
    x = np.arange(len(EXPOSURES))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, ptype in zip(axes, ("exact_prefix", "paraphrased")):
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        b1 = ax.bar(x - width / 2, sub["post_strict_rate"] * 100, width, label="Strict Exact Match", color="#1f4e79")
        b2 = ax.bar(x + width / 2, sub["post_relaxed_rate"] * 100, width, label="Relaxed Prefix Match", color="#70ad47")
        ax.set(title=PROMPT_LABELS[ptype], xlabel="Cumulative Exposures", ylim=(0, 112))
        ax.set_xticks(x, [str(e) for e in EXPOSURES])
        for bar in (*b1, *b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Extraction Success Rate (%)")
    fig.suptitle("Post-Training Extraction Ratio: Strict vs. Relaxed Match", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(errors: pd.DataFrame, out_path) -> None:
    cols = ["Strict Success"] + [label for _, label in ERROR_COLS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for i, (ax, ptype) in enumerate(zip(axes, ("exact_prefix", "paraphrased"))):
        sub = errors[errors["prompt_type"] == ptype].sort_values("cumulative_exposures")
        sns.heatmap(
            sub[cols].to_numpy(), annot=True, fmt="d", cmap="YlGnBu", cbar=(i == 1),
            xticklabels=cols, yticklabels=[f"{e} exp" for e in EXPOSURES], vmin=0, vmax=6, ax=ax,
        )
        ax.set_title(f"{PROMPT_LABELS[ptype]}\n(Multi-label counts out of 6 canaries/group)")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Exposure Group")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_token_bottleneck(summary: pd.DataFrame, out_path) -> None:
    x = np.arange(len(EXPOSURES))
    specs = [
        ("post_w1_ll_mean", "w_1", "o", "#b2182b"),
        ("post_w2_ll_mean", "w_2", "s", "#2166ac"),
        ("post_w3_ll_mean", "w_3", "^", "#4dac26"),
        ("post_w4_ll_mean", "w_4", "D", "#762a83"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for ax, ptype in zip(axes, ("exact_prefix", "paraphrased")):
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        for col, sym, marker, color in specs:
            ax.plot(x, sub[col], marker=marker, lw=2.0, color=color, label=rf"$\ell({sym})$")
        ax.set(title=f"Token-Level Log-Probabilities — {PROMPT_LABELS[ptype]}", xlabel="Cumulative Exposures")
        ax.set_xticks(x, [str(e) for e in EXPOSURES])
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"Mean Post-Training $\log P(w_k \mid X, w_{<k})$")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_training_loss(loss_df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_df["global_step"], loss_df["loss"], color="#1f4e79", lw=1.8)
    ax.set(title="LoRA Training Loss (Full 96-Token Causal LM Loss)", xlabel="Optimizer Step", ylabel="Cross-Entropy Loss")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

    df_all = pd.read_csv(config.RESULTS_DIR / "eval_all_96.csv")
    df_paired = pd.read_csv(config.RESULTS_DIR / "eval_paired_48.csv")
    loss_path = config.RESULTS_DIR / "training_loss.csv"
    loss_df = pd.read_csv(loss_path) if loss_path.exists() else None

    summary = build_summary(df_paired)
    errors = build_error_table(df_paired)
    hyp = check_hypotheses(df_all, df_paired)

    summary.to_csv(config.RESULTS_DIR / "summary_by_group.csv", index=False)
    errors.to_csv(config.RESULTS_DIR / "errors_multilabel_by_group.csv", index=False)
    hyp.to_csv(config.RESULTS_DIR / "hypotheses_h1_h5.csv", index=False)

    plot_predictability(summary, config.PLOTS_DIR / "fig1_predictability_curves.png")
    plot_extraction(summary, config.PLOTS_DIR / "fig2_extraction_rates.png")
    plot_error_heatmap(errors, config.PLOTS_DIR / "fig3_error_taxonomy_multilabel.png")
    plot_token_bottleneck(summary, config.PLOTS_DIR / "fig4_token_level_bottleneck.png")
    if loss_df is not None and not loss_df.empty:
        plot_training_loss(loss_df, config.PLOTS_DIR / "fig5_training_loss.png")

    cols = ["prompt_type", "cumulative_exposures", "pre_ll_mean", "post_ll_mean", "net_delta_ll_mean", "post_strict_rate", "post_relaxed_rate", "post_w1_rank_median"]
    print("\n=== GROUP SUMMARY ===\n", summary[cols].to_string(index=False))
    print("\n=== MULTI-LABEL ERRORS ===\n", errors.to_string(index=False))
    print("\n=== HYPOTHESES (H1-H5) ===\n", hyp.to_string(index=False))
    print(f"\nSaved all figures to: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
