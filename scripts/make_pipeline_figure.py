"""Standalone generator for the SafeSAE-VLA pipeline figure (paper Fig. 1).

Visually encodes the paper's thesis: a single inspectable SAE substrate over the
VLA residual stream feeds both a *safety-by-practice* lane (calibrated runtime
monitor) and a *safety-by-design* lane (representational audit + intervention).

Writes figure1_architecture.{pdf,png} into results/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# ----------------------------------------------------------------------------
# Palette (muted, modern, print-safe)
# ----------------------------------------------------------------------------
C = {
    "sim":      "#6BA4D6",   # light blue
    "model":    "#2F5C9E",   # deep blue
    "hooks":    "#4F9D69",   # green
    "sae":      "#B23A48",   # hero coral-red (the substrate)
    "monitor":  "#6F5BA8",   # indigo  (practice)
    "alert":    "#D98A3D",   # amber   (practice out)
    "audit":    "#C9A227",   # gold    (design)
    "feedback": "#4F9D69",   # green   (design out)
}
PRACTICE_BG, PRACTICE_EDGE = "#EAF2FB", "#A9C9E8"
DESIGN_BG,   DESIGN_EDGE   = "#FBF4E1", "#E6D29A"
INK = "#1b1b1b"
ARROW = "#3a3a3a"

BOLD = FontProperties(weight="bold")


def shadow_box(ax, x, y, w, h, text, color, fontsize=10.0, text_color="white",
               radius=0.018, lw=1.3, edge="#2a2a2a", z=3):
    """A rounded box with a soft drop shadow and centred bold label."""
    bs = f"round,pad=0.012,rounding_size={radius}"
    # shadow
    ax.add_patch(FancyBboxPatch(
        (x + 0.005, y - 0.009), w, h, boxstyle=bs,
        facecolor="#000000", edgecolor="none", alpha=0.12, zorder=z - 1,
        mutation_aspect=1.0))
    # body
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=bs,
        facecolor=color, edgecolor=edge, linewidth=lw, zorder=z,
        mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=z + 1,
            linespacing=1.15)
    return (x + w / 2, y + h / 2)


def panel(ax, x, y, w, h, label, face, edge, label_color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=1.2, zorder=1))
    ax.text(x + 0.016, y + h - 0.028, label, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=label_color, zorder=2,
            fontstyle="italic")


def arrow(ax, p1, p2, color=ARROW, lw=2.0, style="arc3,rad=0.0", z=5,
          scale=15, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, connectionstyle=style, arrowstyle="-|>",
        mutation_scale=scale, lw=lw, color=color, zorder=z,
        shrinkA=2, shrinkB=2, capstyle="round", linestyle=ls))


def seg(ax, p1, p2, color=ARROW, lw=2.0, z=4, ls="-"):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
            zorder=z, solid_capstyle="round", linestyle=ls)


def make(out_base: Path):
    fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- title ----
    ax.text(0.5, 0.965, "SafeSAE-VLA", ha="center", va="center",
            fontsize=16, fontweight="bold", color=INK)
    ax.text(0.5, 0.915,
            "one inspectable substrate, read from both sides of safety",
            ha="center", va="center", fontsize=9.5, fontstyle="italic",
            color="#666666")

    # ---- two paradigm panels (right) ----
    panel(ax, 0.605, 0.560, 0.385, 0.265, "SAFETY BY PRACTICE  ·  deployment-time",
          PRACTICE_BG, PRACTICE_EDGE, "#2F5C9E")
    panel(ax, 0.605, 0.285, 0.385, 0.265, "SAFETY BY DESIGN  ·  source-time",
          DESIGN_BG, DESIGN_EDGE, "#9A7B00")

    # ---- left input column ----
    p_model = shadow_box(ax, 0.014, 0.555, 0.130, 0.150, "OpenVLA 7B",
                         C["model"], fontsize=10.5)
    p_sim = shadow_box(ax, 0.014, 0.330, 0.130, 0.140, "LIBERO\nSimulator",
                       C["sim"], fontsize=9.8, text_color="white")
    ax.text(0.079, 0.745, "rollouts", ha="center", fontsize=7.6,
            color="#777777", fontstyle="italic")

    # ---- hooks (layer label sits INSIDE the box) ----
    shadow_box(ax, 0.196, 0.545, 0.144, 0.170, "", C["hooks"])
    ax.text(0.268, 0.652, "Activation\nHooks", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="white", zorder=5,
            linespacing=1.1)
    ax.text(0.268, 0.578, "L16 · L20 · L24", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#eaf5ee", zorder=5)

    # ---- SAE substrate (hero, taller, bridges both lanes) ----
    shadow_box(ax, 0.392, 0.460, 0.165, 0.330, "", C["sae"], lw=1.8,
               edge="#7a1f29", radius=0.022)
    ax.text(0.4745, 0.650, "BatchTopK\nSAE", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white", zorder=5,
            linespacing=1.1)
    ax.text(0.4745, 0.560, "sparse substrate", ha="center", va="center",
            fontsize=7.8, color="#f3d6da", fontstyle="italic", zorder=5)

    # ---- practice lane ----  (centres at y=0.6675; wide gap for clean arrow)
    shadow_box(ax, 0.622, 0.600, 0.150, 0.135,
               "Calibrated\nMonitor", C["monitor"], fontsize=9.2)
    shadow_box(ax, 0.828, 0.600, 0.150, 0.135,
               "Alert /\nAbstain", C["alert"], fontsize=9.2)

    # ---- design lane ----  (centres at y=0.3925)
    shadow_box(ax, 0.622, 0.325, 0.150, 0.135,
               "Feature Audit\n& Intervene", C["audit"],
               fontsize=8.6, text_color=INK, edge="#9a7b00")
    shadow_box(ax, 0.828, 0.325, 0.150, 0.135,
               "Design\nFeedback", C["feedback"], fontsize=9.2)

    # ---- forward flow (clean, orthogonal) ----
    arrow(ax, (0.079, 0.470), (0.079, 0.555))            # sim -> model
    arrow(ax, (0.144, 0.630), (0.196, 0.630))            # model -> hooks
    arrow(ax, (0.340, 0.630), (0.392, 0.630))            # hooks -> sae
    # SAE forks to both lanes via a clean bus
    seg(ax, (0.557, 0.625), (0.585, 0.625))              # trunk out of SAE
    seg(ax, (0.585, 0.3925), (0.585, 0.6675))            # vertical riser
    arrow(ax, (0.585, 0.6675), (0.620, 0.6675))          # bus -> monitor
    arrow(ax, (0.585, 0.3925), (0.620, 0.3925))          # bus -> audit
    # within lanes (now with a generous gap)
    arrow(ax, (0.772, 0.6675), (0.828, 0.6675), scale=15)   # monitor -> alert
    arrow(ax, (0.772, 0.3925), (0.828, 0.3925), scale=15)   # audit -> feedback
    # design feedback returns toward training (dashed orthogonal bus, below)
    fb = "#9a7b00"
    dash = (0, (5, 3))
    seg(ax, (0.903, 0.325), (0.903, 0.180), color=fb, lw=1.5, z=2, ls=dash)
    seg(ax, (0.903, 0.180), (0.079, 0.180), color=fb, lw=1.5, z=2, ls=dash)
    arrow(ax, (0.079, 0.180), (0.079, 0.327), color=fb, lw=1.5, z=2,
          scale=13, ls=dash)
    ax.text(0.49, 0.180, "design feedback closes the loop into training",
            ha="center", va="center", fontsize=7.6, color=fb, fontstyle="italic",
            zorder=3, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor="none"))

    # ---- hazard category chips ----
    ax.text(0.50, 0.135, "Hazard categories audited", ha="center", fontsize=9,
            fontweight="bold", color="#555555")
    cats = ["collision", "force", "boundary", "speed", "drop"]
    n = len(cats)
    x0, x1 = 0.165, 0.835
    for i, cat in enumerate(cats):
        xp = x0 + i * (x1 - x0) / (n - 1)
        ax.text(xp, 0.062, cat, ha="center", va="center", fontsize=8.6,
                color="#333333", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.42", facecolor="#f4f4f6",
                          edgecolor="#c4c4cc", linewidth=1.0))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)
    print("wrote", out_base.with_suffix(".pdf"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make(root / "results" / "figures" / "figure1_architecture")
