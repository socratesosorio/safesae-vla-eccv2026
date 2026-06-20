"""Generate compact paper figures from final robustness CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robustness_dir", type=str, default="logs/progress_feature_robustness")
    p.add_argument("--output_dir", type=str, default="paper/figures")
    p.add_argument("--top_n", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    robustness_dir = Path(args.robustness_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch = pd.read_csv(robustness_dir / "class_mean_patch_directionality.csv")
    prevalence = pd.read_csv(robustness_dir / "top_feature_activation_prevalence.csv").head(args.top_n)

    top = patch[patch["condition"] == "top20_low_samples_to_high_class_mean"].iloc[0]
    random = patch[patch["condition"].str.startswith("random20_low_samples_to_high_class_mean_trial")]
    random_deltas = random["mean_progress_logit_delta"].to_numpy()
    p_value = (np.sum(random_deltas >= float(top["mean_progress_logit_delta"])) + 1) / (len(random_deltas) + 1)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    parts = ax.violinplot([random_deltas], positions=[0], widths=0.65, showmeans=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9aa0a6")
        body.set_edgecolor("#5f6368")
        body.set_alpha(0.55)
    ax.scatter(np.zeros_like(random_deltas), random_deltas, s=8, color="#5f6368", alpha=0.45, linewidths=0)
    ax.scatter([1], [top["mean_progress_logit_delta"]], s=60, color="#0072B2", zorder=4)
    ax.errorbar(
        [1],
        [top["mean_progress_logit_delta"]],
        yerr=[[top["mean_progress_logit_delta"] - top.get("mean_progress_logit_delta_ci95_low", top["mean_progress_logit_delta"])],
              [top.get("mean_progress_logit_delta_ci95_high", top["mean_progress_logit_delta"]) - top["mean_progress_logit_delta"]]],
        color="#0072B2",
        capsize=3,
        lw=1.2,
        zorder=3,
    )
    ax.axhline(0, color="#202124", lw=0.8, alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nmatched", "Top-20\nfeatures"])
    ax.set_ylabel("Mean progress-logit shift")
    ax.set_title("Class-mean feature patch")
    ax.text(
        0.02,
        0.98,
        f"$p={p_value:.3f}$\n{top['frac_progress_logit_increased'] * 100:.1f}% increase",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    y = np.arange(len(prevalence))[::-1]
    low = prevalence["active_rate_low"].to_numpy() * 100
    high = prevalence["active_rate_high"].to_numpy() * 100
    labels = [f"#{int(r.rank)} f{int(r.feature_idx)}" for r in prevalence.itertuples()]

    for yi, lo, hi in zip(y, low, high):
        ax.plot([lo, hi], [yi, yi], color="#dadce0", lw=1.2, zorder=1)
    ax.scatter(low, y, color="#9aa0a6", s=24, zorder=2)
    ax.scatter(high, y, color="#009E73", s=24, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Activation rate in class (%)")
    ax.set_title("Top-feature prevalence")
    ax.set_xlim(0, max(75, float(max(low.max(), high.max()) + 5)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e8eaed", lw=0.6)

    fig.tight_layout(w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"figure14_intervention_prevalence.{ext}", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
