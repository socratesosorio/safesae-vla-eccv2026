"""Figure 1 for the ECCV 2026 camera-ready 'SafeSAE-VLA' paper.

Reuses the polished rounded-shadow-box aesthetic but depicts THIS paper's actual
progress-analysis pipeline (progress-only framing, no safety-deployment / hazard
content). Writes directly to the camera-ready figures/ directory.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# blue, green, coral, indigo, gold
C = ["#2F5C9E", "#4F9D69", "#B23A48", "#6F5BA8", "#C0872E"]
EDGE = ["#1f3f70", "#356b48", "#7a1f29", "#4b3c78", "#8a5f1f"]
ARROW = "#3a3a3a"


def shadow_box(ax, x, y, w, h, color, edge, lines, fs_main=10.5, fs_sub=6.7,
               sub=None):
    bs = "round,pad=0.010,rounding_size=0.018"
    ax.add_patch(FancyBboxPatch((x + 0.004, y - 0.012), w, h, boxstyle=bs,
                                facecolor="#000000", edgecolor="none",
                                alpha=0.12, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, facecolor=color,
                                edgecolor=edge, linewidth=1.4, zorder=3))
    cy = y + h / 2 + (0.055 if sub else 0)
    ax.text(x + w / 2, cy, lines, ha="center", va="center", fontsize=fs_main,
            fontweight="bold", color="white", zorder=4, linespacing=1.12)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.11, sub, ha="center", va="center",
                fontsize=fs_sub, color="#f2f2f2", zorder=4, fontstyle="italic")


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=15, lw=2.0, color=ARROW,
                                 zorder=5, shrinkA=1, shrinkB=1,
                                 capstyle="round"))


def make(out_base: Path):
    fig, ax = plt.subplots(figsize=(9.2, 2.35), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ("Rollouts", "OpenVLA · LIBERO"),
        ("Progress\nRelabel", "suite-norm quartiles"),
        ("BatchTopK\nSAE  (L20)", "d_sae 16,384 · k 32"),
        ("Differential\nRanking", "MW · BH-FDR"),
        ("Monitor &\nIntervention", "0.918 AUROC"),
    ]

    n = len(boxes)
    margin = 0.045          # keep boxes off the left/right edges (avoid clipping at \linewidth)
    w, h = 0.155, 0.44
    y = 0.30
    gap = (1.0 - 2 * margin - n * w) / (n - 1)
    xs = [margin + i * (w + gap) for i in range(n)]

    ax.text(0.5, 0.94, "SafeSAE-VLA: Progress Analysis Pipeline", ha="center",
            va="center", fontsize=12.5, fontweight="bold", color="#1b1b1b")

    for i, (label, sub) in enumerate(boxes):
        shadow_box(ax, xs[i], y, w, h, C[i], EDGE[i], label, sub=sub)
        if i < n - 1:
            arrow(ax, xs[i] + w, xs[i + 1], y + h / 2)

    ax.text(0.5, 0.05,
            "Action-token residual activations are SAE-encoded, ranked by progress, "
            "and read out by lightweight monitors; interventions are task-local diagnostics.",
            ha="center", va="center", fontsize=7.2, fontstyle="italic",
            color="#666666", wrap=True)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)
    print("wrote", out_base.with_suffix(".pdf"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make(root / "paper_eccv_cr_matched" / "figures" / "figure1_architecture")
