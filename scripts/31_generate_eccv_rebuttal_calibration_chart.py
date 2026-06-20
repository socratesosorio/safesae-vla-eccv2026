"""Generate a compact proxy/safety calibration chart for the ECCV rebuttal."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path("rebuttal-template-Latest/rebuttal_calibration.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.35), gridspec_kw={"width_ratios": [1.0, 1.1]})

    ax = axes[0]
    names = ["final\ndistance", "episode\nsuccess"]
    vals = [-0.926, 0.012]
    colors = ["#0072B2", "#9aa0a6"]
    ax.bar(np.arange(2), vals, color=colors, width=0.58)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_ylim(-1.0, 0.25)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(names)
    ax.set_ylabel("Spearman rho")
    ax.set_title("A. Proxy audit")
    ax.text(1, 0.08, "AUROC\n0.507", ha="center", va="bottom", fontsize=6.2)

    ax = axes[1]
    methods = ["SAE", "Raw\nMLP"]
    auroc = [0.632, 0.640]
    false_alarm = [0.009, 0.079]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, auroc, width=width, color="#009E73", label="AUROC")
    ax.bar(x + width / 2, false_alarm, width=width, color="#D55E00", label="false alarm")
    ax.set_ylim(0, 0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title("B. Safety stress test")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8)
    for xi, val in zip(x - width / 2, auroc):
        ax.text(xi, val + 0.025, f"{val:.3f}", ha="center", fontsize=5.8)
    for xi, val in zip(x + width / 2, false_alarm):
        ax.text(xi, val + 0.025, f"{val:.3f}", ha="center", fontsize=5.8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8e8e8", lw=0.5)

    fig.tight_layout(pad=0.25, w_pad=0.6)
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
