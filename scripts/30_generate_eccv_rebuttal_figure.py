"""Generate a compact visual summary for the ECCV rebuttal."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    out = Path("rebuttal-template-Latest/rebuttal_summary.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv("logs/eccv_rebuttal_checks/rebuttal_progress_baselines_and_splits.csv")
    q = metrics[metrics["scheme"] == "quartile"].set_index("method")
    patch = pd.read_csv("logs/progress_feature_robustness/class_mean_patch_directionality.csv")
    patch_summary = json.loads(Path("logs/progress_feature_robustness/progress_feature_robustness_summary.json").read_text())
    semantic = pd.read_csv("logs/eccv_success_labeled_baseline_audit_after676838/success_labeled_baseline_results.csv")
    top = patch[patch["condition"] == "top20_low_samples_to_high_class_mean"].iloc[0]
    rand = patch[patch["condition"] == "random20_low_samples_to_high_class_mean_mean"].iloc[0]

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(3.25, 3.45))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.12, 0.95, 0.92], hspace=1.05, wspace=0.52)
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    # Panel A: reviewer-requested raw baselines vs SAE.
    ax = axes[0]
    methods = ["Raw activation LR", "Raw activation MLP", "SAE LR", "Top-20 SAE LR"]
    labels = ["Raw LR", "Raw MLP", "SAE LR", "Top-20 SAE"]
    colors = ["#8c8c8c", "#b0b0b0", "#0072B2", "#009E73"]
    vals = np.array([q.loc[m, "auroc"] for m in methods])
    lo = np.array([q.loc[m, "auroc_ci95_low"] for m in methods])
    hi = np.array([q.loc[m, "auroc_ci95_high"] for m in methods])
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=0.58)
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor="#222222", lw=0.8, capsize=2)
    ax.set_xlim(0.88, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AUROC")
    ax.set_title("A. Raw baseline check", pad=8)
    for yi, val in zip(y, vals):
        ax.text(min(val + 0.003, 0.996), yi, f"{val:.3f}", va="center", fontsize=6.2)

    # Panel B: layer sweep.
    ax = axes[1]
    layers = ["L16", "L20", "L24"]
    layer_vals = [0.884, 0.896, 0.871]
    ax.plot(layers, layer_vals, marker="o", color="#CC79A7", lw=1.5)
    ax.set_ylim(0.84, 0.91)
    ax.set_ylabel("AUROC")
    ax.set_title("B. Layer sweep", pad=8)
    ax.text(0.02, 0.05, "Jaccard <= 0.03", transform=ax.transAxes, fontsize=6.5)

    # Panel C: feature-setting sanity check.
    ax = axes[2]
    vals = np.array([float(rand["mean_progress_logit_delta"]), float(top["mean_progress_logit_delta"])])
    lo = np.array([
        float(patch_summary["patch_random_mean_logit_delta_ci95_low"]),
        float(patch_summary["patch_top20_mean_logit_delta_ci95_low"]),
    ])
    hi = np.array([
        float(patch_summary["patch_random_mean_logit_delta_ci95_high"]),
        float(patch_summary["patch_top20_mean_logit_delta_ci95_high"]),
    ])
    x = np.arange(2)
    ax.bar(x, vals, color=["#9aa0a6", "#D55E00"], width=0.55)
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="#222222", lw=0.8, capsize=2)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random", "Top-20"])
    ax.set_ylabel("Logit shift")
    ax.set_title("C. Feature setting", pad=8)
    ax.text(0.05, 0.88, "p=0.005", transform=ax.transAxes, fontsize=6.5)

    # Panel D: progress proxy audit.
    ax = axes[3]
    proxy_names = ["final\ndistance", "episode\nsuccess"]
    proxy_vals = [-0.926, 0.012]
    ax.bar(np.arange(2), proxy_vals, color=["#0072B2", "#9aa0a6"], width=0.58)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_ylim(-1.0, 0.25)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(proxy_names)
    ax.set_ylabel("rho")
    ax.set_title("D. Proxy audit", pad=8)
    ax.text(1, 0.08, "AUC\n0.507", ha="center", va="bottom", fontsize=5.8)

    # Panel E: real success/object-state metadata. This addresses the reviewer
    # concern that the submitted progress target was geometric.
    ax = axes[4]
    methods = ["submitted_top20_sae", "geometry_only", "full_sae_lr", "raw_lr"]
    method_labels = ["Top-20", "Geom.", "Full SAE", "Raw"]
    colors = ["#c7c7c7", "#0072B2", "#009E73", "#6b6b6b"]
    vals = np.array([float(semantic[(semantic["target"] == "success") & (semantic["method"] == method)]["auroc"].iloc[0]) for method in methods])
    x = np.arange(len(methods))
    ax.bar(x, vals, color=colors, width=0.58)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.014, f"{val:.2f}", ha="center", va="bottom", fontsize=5.7)
    ax.set_ylim(0.35, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=22, ha="right")
    ax.tick_params(axis="x", labelsize=5.6, pad=1)
    ax.set_ylabel("AUROC")
    ax.set_title("E. Semantic success", pad=8)
    ax.text(0.02, 0.05, "n=120", transform=ax.transAxes, fontsize=6.4)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.14, right=0.98, hspace=1.05, wspace=0.52)
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
