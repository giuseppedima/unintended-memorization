"""Dedicated plotting functions for the 5 empirical study figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXPOSURES = [0, 2, 10, 40]
PROMPT_LABELS = {
    "exact_prefix": "Exact Training Prefix",
    "paraphrased": "Paraphrased Question",
}
ERROR_CATEGORY_LABELS = [
    "Strict Success",
    "Non-stop",
    "Preamble",
    "Partial Memorization",
    "Out of Order",
    "Wrong Case",
    "Full Miss",
]


def _apply_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)


def plot_predictability_curves(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 1: Pre vs. Post secret log-likelihood and control-adjusted net gain."""
    _apply_plot_theme()
    x_pos = np.arange(len(EXPOSURES))
    palette = {"exact_prefix": "#1f4e79", "paraphrased": "#c55a11"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ptype, color in palette.items():
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        label = PROMPT_LABELS[ptype]
        axes[0].errorbar(
            x_pos, sub["pre_ll_mean"], yerr=sub["pre_ll_sd"],
            fmt="--o", color=color, alpha=0.55, capsize=4, label=f"{label} (Pre)",
        )
        axes[0].errorbar(
            x_pos, sub["post_ll_mean"], yerr=sub["post_ll_sd"],
            fmt="-s", color=color, linewidth=2.2, capsize=4, label=f"{label} (Post)",
        )
        axes[1].errorbar(
            x_pos, sub["net_delta_ll_mean"], yerr=sub["net_delta_ll_sd"],
            fmt="-s", color=color, linewidth=2.2, capsize=4, label=label,
        )

    axes[0].set_title("Secret Log-Likelihood: Pre vs. Post Fine-Tuning")
    axes[0].set_ylabel("Mean Log-Likelihood ± SD")
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    axes[1].set_title(r"Net Memorization Gain ($\Delta\mathcal{L}_{\mathrm{net}}$)")
    axes[1].set_ylabel("Mean Net Gain ± SD (Control-Adjusted)")

    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(e) for e in EXPOSURES])
        ax.set_xlabel("Cumulative Exposures (2 Epochs)")
        ax.legend(fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_extraction_rates(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 2: Grouped bar chart comparing Strict Exact Match vs. Relaxed Prefix Match."""
    _apply_plot_theme()
    x_pos = np.arange(len(EXPOSURES))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for idx, ptype in enumerate(("exact_prefix", "paraphrased")):
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        ax = axes[idx]
        b1 = ax.bar(
            x_pos - width / 2, sub["post_strict_rate"] * 100, width,
            label="Strict Exact Match", color="#1f4e79",
        )
        b2 = ax.bar(
            x_pos + width / 2, sub["post_relaxed_rate"] * 100, width,
            label="Relaxed Prefix Match", color="#70ad47",
        )
        ax.set_title(PROMPT_LABELS[ptype])
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(e) for e in EXPOSURES])
        ax.set_xlabel("Cumulative Exposures")
        ax.set_ylim(0, 112)
        for bars in (b1, b2):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 2,
                    f"{height:.0f}%", ha="center", va="bottom", fontsize=8,
                )
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Extraction Success Rate (%)")
    fig.suptitle("Post-Training Extraction Ratio: Strict vs. Relaxed Match", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_taxonomy_heatmap(error_table: pd.DataFrame, output_path: Path) -> None:
    """Figure 3: Multi-label error taxonomy heatmap across the 6 slide categories."""
    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for idx, ptype in enumerate(("exact_prefix", "paraphrased")):
        sub = error_table[error_table["prompt_type"] == ptype].sort_values("cumulative_exposures")
        matrix = sub[ERROR_CATEGORY_LABELS].to_numpy()
        ax = axes[idx]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=(idx == 1),
            xticklabels=ERROR_CATEGORY_LABELS,
            yticklabels=[f"{e} exp" for e in EXPOSURES],
            vmin=0,
            vmax=6,
            ax=ax,
        )
        ax.set_title(f"{PROMPT_LABELS[ptype]}\n(Multi-label counts out of 6 canaries/group)")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Exposure Group")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_token_level_bottleneck(summary: pd.DataFrame, output_path: Path) -> None:
    """Figure 4: Token-level post-training log-probabilities l(w1)...l(w4)."""
    _apply_plot_theme()
    x_pos = np.arange(len(EXPOSURES))
    word_specs = [
        ("post_w1_ll_mean", "w_1", "o", "#b2182b"),
        ("post_w2_ll_mean", "w_2", "s", "#2166ac"),
        ("post_w3_ll_mean", "w_3", "^", "#4dac26"),
        ("post_w4_ll_mean", "w_4", "D", "#762a83"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for idx, ptype in enumerate(("exact_prefix", "paraphrased")):
        sub = summary[summary["prompt_type"] == ptype].sort_values("cumulative_exposures")
        ax = axes[idx]
        for col, sym, marker, color in word_specs:
            ax.plot(
                x_pos, sub[col], marker=marker, linewidth=2.0,
                color=color, label=rf"$\ell({sym})$",
            )
        ax.set_title(f"Token-Level Log-Probabilities — {PROMPT_LABELS[ptype]}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(e) for e in EXPOSURES])
        ax.set_xlabel("Cumulative Exposures")
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"Mean Post-Training $\log P(w_k \mid X, w_{<k})$")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_training_loss(loss_df: pd.DataFrame, output_path: Path) -> None:
    """Figure 5: Step-by-step full 96-token Causal LM training loss."""
    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_df["global_step"], loss_df["loss"], color="#1f4e79", linewidth=1.8)
    ax.set_title("LoRA Training Loss (Full 96-Token Causal LM Loss)")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_all_plots(
    summary: pd.DataFrame,
    error_table: pd.DataFrame,
    loss_df: pd.DataFrame | None,
    plots_dir: Path,
) -> None:
    """Generate and save all empirical study figures into `plots_dir`."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_predictability_curves(summary, plots_dir / "fig1_predictability_curves.png")
    plot_extraction_rates(summary, plots_dir / "fig2_extraction_rates.png")
    plot_error_taxonomy_heatmap(error_table, plots_dir / "fig3_error_taxonomy_multilabel.png")
    plot_token_level_bottleneck(summary, plots_dir / "fig4_token_level_bottleneck.png")
    if loss_df is not None and not loss_df.empty:
        plot_training_loss(loss_df, plots_dir / "fig5_training_loss.png")
