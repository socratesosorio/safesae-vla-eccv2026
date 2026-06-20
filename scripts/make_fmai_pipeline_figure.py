"""Professional Figure 1 for the FMAI/ICML 'Sparse Trace Diagnostics' paper.

A clean horizontal pipeline: rollouts -> progress/safety relabeling ->
sparse-feature analysis -> trace diagnostics. Saved as a SEPARATE file so the
RSS paper's figure1_architecture.pdf is untouched.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = ["#2F5C9E", "#4F9D69", "#B23A48", "#6F5BA8"]   # blue, green, coral, indigo
EDGE = ["#1f3f70", "#356b48", "#7a1f29", "#4b3c78"]
ARROW = "#3a3a3a"


def shadow_box(ax, x, y, w, h, color, edge, lines, fs_main=11, fs_sub=7.6,
               sub=None):
    bs = "round,pad=0.012,rounding_size=0.02"
    ax.add_patch(FancyBboxPatch((x + 0.004, y - 0.012), w, h, boxstyle=bs,
                                facecolor="#000000", edgecolor="none",
                                alpha=0.12, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, facecolor=color,
                                edgecolor=edge, linewidth=1.4, zorder=3))
    cy = y + h / 2 + (0.018 if sub else 0)
    ax.text(x + w / 2, cy, lines, ha="center", va="center", fontsize=fs_main,
            fontweight="bold", color="white", zorder=4, linespacing=1.12)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.085, sub, ha="center", va="center",
                fontsize=fs_sub, color="#f2f2f2", zorder=4, fontstyle="italic")


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=16, lw=2.1, color=ARROW,
                                 zorder=5, shrinkA=1, shrinkB=1,
                                 capstyle="round"))


def make(out_base: Path):
    fig, ax = plt.subplots(figsize=(8.4, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ("Rollouts\n750 episodes", "OpenVLA · LIBERO"),
        ("Progress \\&\nSafety Relabel", "suite-norm quartiles"),
        ("Sparse Feature\nAnalysis", "1{,}117/1{,}881 FDR-sig."),
        ("Trace\nDiagnostics", "progress 0.918 AUROC"),
    ]
    # plain text (matplotlib, not LaTeX): fix escapes
    boxes = [
        ("Rollouts\n750 episodes", "OpenVLA · LIBERO"),
        ("Progress &\nSafety Relabel", "suite-norm quartiles"),
        ("Sparse Feature\nAnalysis", "1,117/1,881 FDR-sig."),
        ("Trace\nDiagnostics", "progress 0.918 AUROC"),
    ]

    w, h = 0.205, 0.40
    y = 0.34
    gap = (0.97 - 4 * w) / 3.0
    xs = [0.015 + i * (w + gap) for i in range(4)]

    ax.text(0.5, 0.93, "SAE-VLA Trace-Diagnostics Pipeline", ha="center",
            va="center", fontsize=12.5, fontweight="bold", color="#1b1b1b")

    for i, (label, sub) in enumerate(boxes):
        shadow_box(ax, xs[i], y, w, h, C[i], EDGE[i], label, sub=sub)
        if i < 3:
            arrow(ax, xs[i] + w, xs[i + 1], y + h / 2)

    ax.text(0.5, 0.075,
            "Feature-setting interventions are reported as diagnostics, "
            "not verified policy fixes.",
            ha="center", va="center", fontsize=8.2, fontstyle="italic",
            color="#666666")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.05)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print("wrote", out_base.with_suffix(".pdf"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make(root / "results" / "figures" / "figure1_trace_pipeline")
