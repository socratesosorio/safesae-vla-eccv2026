"""Generate full paper figures and LaTeX-ready tables from SafeSAE-VLA artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.feature_visualization import plot_overlap_heatmap, plot_volcano
from src.data.activation_dataset import ActivationDataset
from src.sae.model import BatchTopKSAE
from src.sae.train_sae import BatchTopKSAE as LegacyBatchTopKSAE
from src.utils.config import load_yaml

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "DejaVu Serif"

SAFETY_CATEGORIES = [
    "collision",
    "excessive_force",
    "boundary_violation",
    "high_approach_speed",
    "object_drop",
]


def tex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SafeSAE-VLA figures")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--paper_dir", type=str, default="paper")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--sae_checkpoint", type=str, default="")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--max_tsne_episodes", type=int, default=300)
    return parser.parse_args()


def save_dual(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_placeholder(base_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=10)
    save_dual(fig, base_path)


def figure_architecture(base_path: Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    colors = {
        "model": "#4C72B0",
        "hooks": "#55A868",
        "sae": "#C44E52",
        "monitor": "#8172B2",
        "clamp": "#CCB974",
        "sim": "#64B5CD",
    }

    def _box(ax, x, y, w, h, text, color, fontsize=9.5):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black",
            linewidth=1.4, alpha=0.85,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white",
        )

    # Row 1: main pipeline (left to right)
    _box(ax, 0.01, 0.55, 0.14, 0.16, "OpenVLA\n7B", colors["model"])
    _box(ax, 0.19, 0.55, 0.18, 0.16, "Activation Hooks\n(L16, L20, L24)", colors["hooks"], fontsize=8.5)
    _box(ax, 0.41, 0.55, 0.15, 0.16, "SAE\nEncoder", colors["sae"])
    _box(ax, 0.01, 0.28, 0.14, 0.16, "LIBERO\nSimulator", colors["sim"])

    # Row 2: downstream diagnostics (branching)
    _box(ax, 0.62, 0.66, 0.18, 0.14, "Offline SAE\nReadout", colors["monitor"], fontsize=8.2)
    _box(ax, 0.62, 0.42, 0.18, 0.14, "Feature-Setting\nDiagnostic", colors["clamp"], fontsize=7.4)
    _box(ax, 0.84, 0.66, 0.15, 0.14, "Prefix\nDiagnostic", "#DD8452", fontsize=8.4)
    _box(ax, 0.84, 0.42, 0.15, 0.14, "Readout\nShift", "#55A868", fontsize=8.4)

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.12,head_length=0.08", lw=1.6, color="#333333")
    arrows = [
        ((0.15, 0.63), (0.19, 0.63)),     # model -> hooks
        ((0.37, 0.63), (0.41, 0.63)),     # hooks -> sae
        ((0.56, 0.66), (0.62, 0.73)),     # sae -> monitor
        ((0.56, 0.60), (0.62, 0.53)),     # sae -> clamping
        ((0.80, 0.73), (0.84, 0.73)),     # readout -> prefix diagnostic
        ((0.80, 0.49), (0.84, 0.49)),     # feature setting -> readout shift
        ((0.08, 0.44), (0.08, 0.55)),     # sim -> model
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    # Bottom annotations
    cats = ["progress", "actions", "prefix risk", "features", "controls"]
    ax.text(0.50, 0.18, "Audited Signals:", ha="center", fontsize=9,
            fontstyle="italic", color="#555555")
    for i, cat in enumerate(cats):
        x_pos = 0.14 + i * 0.19
        ax.text(x_pos, 0.08, cat, ha="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.8))

    ax.set_title("Sparse VLA Progress Analysis Pipeline", fontsize=13, fontweight="bold", pad=12)
    save_dual(fig, base_path)


def figure_roc_panels(roc_csv: Path, per_cat_roc_csv: Path, base_path: Path) -> None:
    if not roc_csv.exists() and not per_cat_roc_csv.exists():
        save_placeholder(base_path, "ROC Curves", "Run monitor evaluation to generate ROC artifacts.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=300)

    if roc_csv.exists():
        roc_df = pd.read_csv(roc_csv)
        for method, group in roc_df.groupby("method"):
            axes[0].plot(group["fpr"], group["tpr"], label=method)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axes[0].set_title("Overall ROC")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].legend(fontsize=7)

    if per_cat_roc_csv.exists():
        cat_df = pd.read_csv(per_cat_roc_csv)
        for cat, group in cat_df.groupby("category"):
            axes[1].plot(group["fpr"], group["tpr"], label=cat)
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axes[1].set_title("Per-Category ROC (SAE LR)")
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].legend(fontsize=7)

    fig.tight_layout()
    save_dual(fig, base_path)


def figure_causal_and_pareto(causal_csv: Path, pareto_csv: Path, base_path: Path) -> None:
    if not causal_csv.exists() and not pareto_csv.exists():
        save_placeholder(base_path, "Causal + Pareto", "Run causal validation and monitor eval to populate.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=300)

    if causal_csv.exists():
        cdf = pd.read_csv(causal_csv)
        if {"num_features", "collision_rate", "scale"}.issubset(cdf.columns):
            sns.lineplot(
                data=cdf,
                x="num_features",
                y="collision_rate",
                hue="scale",
                marker="o",
                palette="colorblind",
                ax=axes[0],
            )
            axes[0].set_title("Collision vs Clamped Features")
            axes[0].set_xlabel("Number of Clamped Features")
            axes[0].set_ylabel("Collision Rate")
        elif {"collision_rate_clamped", "collision_rate_baseline"}.issubset(cdf.columns):
            cr_clamped = float(cdf["collision_rate_clamped"].mean())
            cr_baseline = float(cdf["collision_rate_baseline"].mean())
            rows = [
                {"variant": "clamped", "collision_rate": cr_clamped},
                {"variant": "baseline", "collision_rate": cr_baseline},
            ]
            sdf = pd.DataFrame(rows)
            sns.barplot(data=sdf, x="variant", y="collision_rate", palette="colorblind", ax=axes[0])
            axes[0].set_title("Collision Rate (Clamped vs Baseline)")
            axes[0].set_xlabel("Variant")
            axes[0].set_ylabel("Collision Rate")
            # Annotate null result when both rates are near 1.0
            if cr_clamped > 0.95 and cr_baseline > 0.95:
                axes[0].text(
                    0.5, 0.5,
                    "No effect: 0% baseline\nsuccess rate",
                    ha="center", va="center", fontsize=8,
                    transform=axes[0].transAxes, color="#666666",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
                )

    if pareto_csv.exists():
        pdf = pd.read_csv(pareto_csv)
        if {"success_rate", "safety_violation_rate"}.issubset(pdf.columns):
            axes[1].plot(pdf["success_rate"], pdf["safety_violation_rate"], color="tab:blue", lw=2)
            axes[1].scatter(pdf["success_rate"], pdf["safety_violation_rate"], s=10, alpha=0.6)
            axes[1].set_title("Pareto: Success vs Safety")
            axes[1].set_xlabel("Success Rate")
            axes[1].set_ylabel("Safety Violation Rate")
            # Annotate degenerate pareto (all points at 0% success)
            if float(pdf["success_rate"].max()) < 0.05:
                axes[1].text(
                    0.5, 0.5,
                    "Degenerate: model achieves\n0% task success",
                    ha="center", va="center", fontsize=8,
                    transform=axes[1].transAxes, color="#666666",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
                )

    fig.tight_layout()
    save_dual(fig, base_path)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def resolve_artifact_dir(results_dir: Path, subdir: str) -> Path:
    candidates = [results_dir / subdir, results_dir / "analysis" / subdir]
    if results_dir.name == "analysis":
        candidates.append(results_dir.parent / subdir)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def figure_ablations(ablations_dir: Path, causal_csv: Path, base_path: Path, results_dir: Path | None = None) -> None:
    dict_path = ablations_dir / "dictionary_size.csv"
    layer_path = ablations_dir / "layer_comparison.csv"

    # Try to synthesize ablation data from differential results if CSVs are missing.
    diff_dir = resolve_artifact_dir(results_dir, "differential") if results_dir else ablations_dir

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), dpi=300)

    # Left panel: Layer ablation (significant features per layer)
    layer_data_found = False
    if layer_path.exists():
        ldf = pd.read_csv(layer_path)
        x_col = find_column(ldf, ["layer", "variant"]) or ldf.columns[0]
        y_col = find_column(ldf, ["significant_features", "collision_reduction", "collision_rate"]) or ldf.columns[min(1, len(ldf.columns)-1)]
        sns.barplot(data=ldf, x=x_col, y=y_col, ax=axes[0], palette="colorblind")
        axes[0].set_title("Layer Ablation")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel(y_col)
        layer_data_found = True
    elif diff_dir.exists():
        # Synthesize from differential analysis CSVs
        layer_rows = []
        for layer_num in [16, 20, 24]:
            for prefix in ["openvla_layer", "layer"]:
                csv_path = diff_dir / f"{prefix}{layer_num}_overall.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    sig = int(df["significant"].astype(bool).sum()) if "significant" in df.columns else 0
                    mean_eff = float(df["abs_effect_size"].head(50).mean()) if "abs_effect_size" in df.columns and not df.empty else 0.0
                    layer_rows.append({"layer": f"L{layer_num}", "significant_features": sig, "mean_effect": mean_eff})
                    break
        if layer_rows:
            ldf = pd.DataFrame(layer_rows)
            sns.barplot(data=ldf, x="layer", y="significant_features", ax=axes[0], palette="Blues_d")
            # Add effect size as secondary annotation
            for i, row in ldf.iterrows():
                axes[0].text(i, row["significant_features"] + 20, f"eff={row['mean_effect']:.2f}",
                           ha="center", fontsize=7, color="#555555")
            axes[0].set_title("Significant Features by Layer")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Significant Features")
            layer_data_found = True
    if not layer_data_found:
        axes[0].text(0.5, 0.5, "No layer comparison data", ha="center", va="center", fontsize=9, color="#888888")
        axes[0].axis("off")

    # Right panel: Dictionary size ablation
    dict_data_found = False
    if dict_path.exists():
        ddf = pd.read_csv(dict_path)
        x_col = find_column(ddf, ["dict_size", "dictionary_size", "d_sae"]) or ddf.columns[0]
        y_col = find_column(ddf, ["auroc", "roc_auc", "significant_features"]) or ddf.columns[min(1, len(ddf.columns) - 1)]
        sns.barplot(data=ddf, x=x_col, y=y_col, ax=axes[1], palette="Greens_d")
        axes[1].set_title("Dictionary Size Ablation")
        axes[1].set_xlabel("Dictionary Size")
        axes[1].set_ylabel(y_col)
        dict_data_found = True
    elif diff_dir.exists():
        # Compare d_sae=16384 vs d_sae=32768 at layer 20
        dict_rows = []
        for prefix, d_sae_label in [("openvla_layer20", "16K"), ("openvla_d32768_layer20", "32K"), ("layer20", "16K")]:
            csv_path = diff_dir / f"{prefix}_overall.csv"
            if csv_path.exists() and not any(r["d_sae"] == d_sae_label for r in dict_rows):
                df = pd.read_csv(csv_path)
                sig = int(df["significant"].astype(bool).sum()) if "significant" in df.columns else 0
                mean_eff = float(df["abs_effect_size"].head(50).mean()) if "abs_effect_size" in df.columns and not df.empty else 0.0
                dict_rows.append({"d_sae": d_sae_label, "significant_features": sig, "mean_effect": mean_eff})
        if dict_rows:
            ddf = pd.DataFrame(dict_rows)
            sns.barplot(data=ddf, x="d_sae", y="significant_features", ax=axes[1], palette="Greens_d")
            for i, row in ddf.iterrows():
                axes[1].text(i, row["significant_features"] + 20, f"eff={row['mean_effect']:.2f}",
                           ha="center", fontsize=7, color="#555555")
            axes[1].set_title("Dict Size Ablation (L20)")
            axes[1].set_xlabel("Dictionary Size (d_sae)")
            axes[1].set_ylabel("Significant Features")
            dict_data_found = True
    if not dict_data_found:
        if causal_csv.exists():
            cdf = pd.read_csv(causal_csv)
            if {"num_features", "collision_rate"}.issubset(cdf.columns):
                cdf = cdf.sort_values("num_features")
                sns.lineplot(data=cdf, x="num_features", y="collision_rate", marker="o", ax=axes[1], color="tab:orange")
                axes[1].set_title("Proxy Ablation (Clamp Count)")
                axes[1].set_xlabel("Num Features")
                axes[1].set_ylabel("Collision Rate")
        else:
            axes[1].text(0.5, 0.5, "No dictionary size data", ha="center", va="center", fontsize=9, color="#888888")
            axes[1].axis("off")

    fig.tight_layout()
    save_dual(fig, base_path)


def _load_overall_csv(diff_dir: Path, stem: str) -> pd.DataFrame:
    path = diff_dir / stem
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def figure_cross_model_structure(results_dir: Path, base_path: Path) -> None:
    diff_dir = resolve_artifact_dir(results_dir, "differential")
    openvla_files = {
        16: _load_overall_csv(diff_dir, "openvla_layer16_overall.csv"),
        20: _load_overall_csv(diff_dir, "openvla_layer20_overall.csv"),
        24: _load_overall_csv(diff_dir, "openvla_layer24_overall.csv"),
    }
    pi0_files = {
        9: _load_overall_csv(diff_dir, "pi0_layer9_overall.csv"),
        11: _load_overall_csv(diff_dir, "pi0_layer11_overall.csv"),
        14: _load_overall_csv(diff_dir, "pi0_layer14_overall.csv"),
    }

    if all(df.empty for df in openvla_files.values()) and all(df.empty for df in pi0_files.values()):
        save_placeholder(base_path, "Cross-Model Structure", "No OpenVLA/pi0 differential outputs found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=300)
    positions = ["Early\n(~50%)", "Mid\n(~63%)", "Late\n(~75%)"]
    ov_counts = [int(openvla_files[k]["significant"].astype(bool).sum()) if not openvla_files[k].empty and "significant" in openvla_files[k] else 0 for k in [16, 20, 24]]
    p0_counts = [int(pi0_files[k]["significant"].astype(bool).sum()) if not pi0_files[k].empty and "significant" in pi0_files[k] else 0 for k in [9, 11, 14]]
    x = np.arange(len(positions))
    width = 0.36
    ax1.bar(x - width / 2, ov_counts, width=width, label="OpenVLA", color="#1f77b4")
    ax1.bar(x + width / 2, p0_counts, width=width, label="pi0", color="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.set_ylabel("Significant Features")
    ax1.set_title("Safety Feature Distribution by Layer")
    ax1.legend(fontsize=7)

    ov_eff = []
    p0_eff = []
    for df in openvla_files.values():
        if not df.empty and "abs_effect_size" in df:
            ov_eff.extend(df["abs_effect_size"].head(100).tolist())
        elif not df.empty and "effect_size" in df:
            ov_eff.extend(np.abs(df["effect_size"].head(100)).tolist())
    for df in pi0_files.values():
        if not df.empty and "abs_effect_size" in df:
            p0_eff.extend(df["abs_effect_size"].head(100).tolist())
        elif not df.empty and "effect_size" in df:
            p0_eff.extend(np.abs(df["effect_size"].head(100)).tolist())
    sns.violinplot(data=[ov_eff or [0.0], p0_eff or [0.0]], ax=ax2, palette=["#1f77b4", "#ff7f0e"])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["OpenVLA", "pi0"])
    ax2.set_ylabel("|Effect Size|")
    ax2.set_title("Top Safety Feature Strength")
    fig.tight_layout()
    save_dual(fig, base_path)


def figure_cross_model_behavior(results_dir: Path, base_path: Path) -> None:
    monitor_dir = resolve_artifact_dir(results_dir, "monitor")
    causal_dir = resolve_artifact_dir(results_dir, "causal")

    def _resolve_first(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    ov_auroc = _resolve_first(
        [
            monitor_dir / "layer20_per_category_auroc.csv",
            monitor_dir / "openvla_layer20_per_category_auroc.csv",
        ]
    )
    p0_auroc = _resolve_first(
        [
            monitor_dir / "layer11_per_category_auroc.csv",
            monitor_dir / "pi0_layer11_per_category_auroc.csv",
        ]
    )
    ov_clamp = _resolve_first(
        [
            causal_dir / "layer20_causal_validation.csv",
            causal_dir / "openvla_layer20_causal_validation.csv",
        ]
    )
    p0_clamp = _resolve_first(
        [
            causal_dir / "layer11_causal_validation.csv",
            causal_dir / "pi0_layer11_causal_validation.csv",
        ]
    )

    if ov_auroc is None and p0_auroc is None and ov_clamp is None and p0_clamp is None:
        save_placeholder(base_path, "Cross-Model Behavior", "No OpenVLA/pi0 monitor or causal outputs found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=300)
    rows = []
    if ov_auroc is not None:
        d = pd.read_csv(ov_auroc)
        if "method" in d.columns:
            d = d[d["method"] == "sae_lr"]
        for _, r in d.iterrows():
            rows.append({"category": r["category"], "model": "OpenVLA", "auroc": float(r["auroc"])})
    if p0_auroc is not None:
        d = pd.read_csv(p0_auroc)
        if "method" in d.columns:
            d = d[d["method"] == "sae_lr"]
        for _, r in d.iterrows():
            rows.append({"category": r["category"], "model": "pi0", "auroc": float(r["auroc"])})
    if rows:
        rdf = pd.DataFrame(rows)
        # Shorten category names for readability
        label_map = {
            "collision": "collision",
            "excessive_force": "exc. force",
            "boundary_violation": "boundary",
            "high_approach_speed": "high speed",
            "object_drop": "obj. drop",
        }
        rdf["category"] = rdf["category"].map(lambda c: label_map.get(c, c))
        sns.barplot(data=rdf, x="category", y="auroc", hue="model", ax=ax1)
        ax1.tick_params(axis="x", rotation=30)
        plt.setp(ax1.get_xticklabels(), ha="right")
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Per-Category AUROC")

    crows = []
    if ov_clamp is not None:
        d = pd.read_csv(ov_clamp)
        crows.append({"model": "OpenVLA", "baseline_or_clamped": "Top-k", "collision_rate": float(d["collision_rate_clamped"].mean())})
        crows.append({"model": "OpenVLA", "baseline_or_clamped": "Baseline", "collision_rate": float(d["collision_rate_baseline"].mean())})
    if p0_clamp is not None:
        d = pd.read_csv(p0_clamp)
        crows.append({"model": "pi0", "baseline_or_clamped": "Top-k", "collision_rate": float(d["collision_rate_clamped"].mean())})
        crows.append({"model": "pi0", "baseline_or_clamped": "Baseline", "collision_rate": float(d["collision_rate_baseline"].mean())})
    if crows:
        cdf = pd.DataFrame(crows)
        sns.barplot(data=cdf, x="model", y="collision_rate", hue="baseline_or_clamped", ax=ax2)
        # Annotate if all collision rates are near 1.0 (no clamping effect)
        if all(r["collision_rate"] > 0.95 for r in crows):
            ax2.text(
                0.5, 0.5, "No effect:\n0% baseline success",
                ha="center", va="center", fontsize=8,
                transform=ax2.transAxes, color="#666666",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            )
    ax2.set_title("Clamping Collision Rates")
    fig.tight_layout()
    save_dual(fig, base_path)


def load_sae_model(sae_checkpoint: Path, sae_config_path: Path, device: torch.device):
    sae_cfg = load_yaml(sae_config_path)
    sae_block = sae_cfg.get("primary", sae_cfg.get("sae", sae_cfg))
    d_in = int(sae_block.get("d_in", 4096))
    d_sae = int(sae_block.get("d_sae", 16384))
    k = int(sae_block.get("k", 32))

    ckpt = torch.load(str(sae_checkpoint), map_location=device)
    d_in = int(ckpt.get("d_in", d_in))
    d_sae = int(ckpt.get("d_sae", d_sae))
    k = int(ckpt.get("k", k))
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    last_exc: Exception | None = None
    for cls in (BatchTopKSAE, LegacyBatchTopKSAE):
        try:
            model = cls(d_in=d_in, d_sae=d_sae, k=k).to(device)  # type: ignore[call-arg]
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Unable to load SAE checkpoint {sae_checkpoint}: {last_exc}")


def figure_feature_heatmaps(
    data_dir: Path,
    sae_checkpoint: Path,
    sae_config_path: Path,
    overall_csv: Path,
    layer: int,
    base_path: Path,
) -> None:
    if not (data_dir.exists() and sae_checkpoint.exists() and sae_config_path.exists() and overall_csv.exists()):
        save_placeholder(base_path, "Feature Heatmaps", "Pass --data_dir and --sae_checkpoint after analysis.")
        return

    overall_df = pd.read_csv(overall_csv)
    if overall_df.empty:
        save_placeholder(base_path, "Feature Heatmaps", "No differential feature rows found.")
        return

    top_feats = overall_df["feature_idx"].head(4).astype(int).tolist()
    if not top_feats:
        save_placeholder(base_path, "Feature Heatmaps", "No ranked features available.")
        return

    dataset = ActivationDataset(str(data_dir), layer=layer, split="all")
    if len(dataset) < 2:
        save_placeholder(base_path, "Feature Heatmaps", "Need at least two episodes in dataset.")
        return

    safe_idx, unsafe_idx = dataset.get_safe_unsafe_split()
    if not safe_idx or not unsafe_idx:
        # Severity fallback: all episodes have violations.
        # Use lowest/highest violation-count episodes as proxies.
        violation_counts = []
        for i in range(len(dataset)):
            item = dataset[i]
            vcount = int(item["episode_safety_violations"].sum().item())
            violation_counts.append((i, vcount))
        violation_counts.sort(key=lambda x: x[1])
        n = len(violation_counts)
        if n < 4:
            save_placeholder(base_path, "Feature Heatmaps", "Need at least 4 episodes for severity split.")
            return
        # Bottom quartile = "safe" proxy, top quartile = "unsafe" proxy
        low_cut = max(1, n // 4)
        high_cut = max(n - n // 4, low_cut + 1)
        safe_idx = [idx for idx, _ in violation_counts[:low_cut]]
        unsafe_idx = [idx for idx, _ in violation_counts[high_cut:]]
        if not safe_idx or not unsafe_idx:
            save_placeholder(base_path, "Feature Heatmaps", "Could not split episodes by severity.")
            return

    safe_item = dataset[safe_idx[0]]
    unsafe_item = dataset[unsafe_idx[0]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = load_sae_model(sae_checkpoint, sae_config_path, device)

    def traces(item: dict) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            pooled = item["activations"].mean(dim=1).to(device)
            feats = sae.encode(pooled).detach().cpu().numpy()
        mat = feats[:, top_feats].T
        viol = item["safety_labels"].any(dim=2).any(dim=1).numpy().astype(float)
        return mat, viol

    unsafe_mat, unsafe_viol = traces(unsafe_item)
    safe_mat, safe_viol = traces(safe_item)

    # Use shared color scale so both panels are directly comparable
    vmin = min(float(unsafe_mat.min()), float(safe_mat.min()))
    vmax = max(float(unsafe_mat.max()), float(safe_mat.max()))

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.5), dpi=300,
                              gridspec_kw={"height_ratios": [1, 8], "hspace": 0.05})

    def _plot_panel(ax_viol, ax_heat, mat, viol, title):
        # Top strip: violation indicator (thin red bar where violations occur)
        ax_viol.imshow(viol.reshape(1, -1), aspect="auto", cmap="Reds",
                       vmin=0, vmax=1, interpolation="nearest")
        ax_viol.set_xticks([])
        ax_viol.set_yticks([])
        ax_viol.set_ylabel("viol", fontsize=7, rotation=0, labelpad=20, va="center")
        ax_viol.set_title(title, fontsize=10)

        # Main heatmap: feature activations (no violation lines)
        sns.heatmap(mat, cmap="viridis", ax=ax_heat, vmin=vmin, vmax=vmax,
                    cbar_kws={"shrink": 0.8})
        ax_heat.set_xlabel("Timestep")
        ax_heat.set_ylabel("Top Feature")
        # Reduce x-tick density
        n_steps = mat.shape[1]
        tick_positions = np.linspace(0, n_steps - 1, min(6, n_steps)).astype(int)
        ax_heat.set_xticks(tick_positions)
        ax_heat.set_xticklabels(tick_positions, fontsize=8)

    _plot_panel(axes[0, 0], axes[1, 0], unsafe_mat, unsafe_viol, "High-Severity Episode")
    _plot_panel(axes[0, 1], axes[1, 1], safe_mat, safe_viol, "Low-Severity Episode")

    fig.tight_layout()
    save_dual(fig, base_path)


def figure_tsne(
    data_dir: Path,
    sae_checkpoint: Path,
    sae_config_path: Path,
    layer: int,
    max_episodes: int,
    base_path: Path,
) -> None:
    if not (data_dir.exists() and sae_checkpoint.exists() and sae_config_path.exists()):
        save_placeholder(base_path, "t-SNE", "Pass --data_dir and --sae_checkpoint to render embedding.")
        return

    dataset = ActivationDataset(str(data_dir), layer=layer, split="all")
    if len(dataset) < 2:
        save_placeholder(base_path, "t-SNE", "Need at least two episodes in dataset.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = load_sae_model(sae_checkpoint, sae_config_path, device)

    n = min(len(dataset), max_episodes)
    features = []
    violation_counts = []
    suite_labels = []

    for i in range(n):
        item = dataset[i]
        with torch.no_grad():
            pooled = item["activations"].mean(dim=1).to(device)
            acts = sae.encode(pooled)
            ep_feat = acts.mean(dim=0).detach().cpu().numpy()
        features.append(ep_feat)
        # Use total violation count (continuous) instead of binary unsafe label
        vcount = int(item["episode_safety_violations"].sum().item())
        violation_counts.append(vcount)
        # Extract suite name for shape markers if available
        meta = item.get("metadata", {})
        suite_labels.append(str(meta.get("suite", "unknown")))

    x = np.asarray(features)
    y = np.asarray(violation_counts, dtype=np.float32)
    if x.shape[0] < 3:
        save_placeholder(base_path, "t-SNE", "Need at least three episodes for embedding.")
        return

    perplexity = min(30, max(2, (x.shape[0] - 1) // 3))
    emb = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(x)

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=300)
    # Use log scale for violation count to spread the color range
    y_display = np.log1p(y)
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y_display, cmap="YlOrRd", s=14, alpha=0.8, edgecolors="none")
    ax.set_title("t-SNE of Episode SAE Feature Space")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log(1 + violation count)")
    fig.tight_layout()
    save_dual(fig, base_path)


def figure_sparsity_curve(csv_path: Path, base_path: Path) -> None:
    """Figure 10: AUROC vs number of top-k SAE features (log scale)."""
    if not csv_path.exists():
        save_placeholder(base_path, "Sparsity Curve", "Run 08_additional_analyses.py first.")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=300)

    ax.plot(df["k"], df["auroc_overall"], marker="o", linewidth=2.2, label="Overall", color="tab:blue", zorder=5)

    # Only show categories with signal above chance
    show_cats = ["boundary_violation", "collision", "excessive_force"]
    cat_colors = {"collision": "tab:red", "excessive_force": "tab:orange",
                  "boundary_violation": "#2ca02c"}
    label_map = {"collision": "Collision", "excessive_force": "Exc. Force",
                 "boundary_violation": "Boundary Viol."}
    for cat in show_cats:
        col = f"auroc_{cat}"
        if col in df.columns:
            ax.plot(df["k"], df[col], marker="s", linewidth=1.4, alpha=0.8,
                    label=label_map.get(cat, cat), color=cat_colors.get(cat, "gray"), markersize=4)

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Top Features (k)")
    ax.set_ylabel("AUROC")
    ax.set_title("Safety Monitor Performance vs Feature Sparsity")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    save_dual(fig, base_path)


def figure_temporal_patterns(csv_path: Path, base_path: Path) -> None:
    """Figure 11: Mean top-feature activation around boundary violation onset."""
    if not csv_path.exists():
        save_placeholder(base_path, "Temporal Patterns", "Run 08_additional_analyses.py first.")
        return

    df = pd.read_csv(csv_path)

    # Focus on boundary_violation — the only category with a clear temporal signal
    cat_df = df[df["category"] == "boundary_violation"].sort_values("relative_t")
    if cat_df.empty:
        save_placeholder(base_path, "Temporal Patterns", "No boundary violation events found.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

    ax.fill_between(cat_df["relative_t"],
                    cat_df["mean_activation"] - cat_df["std_activation"],
                    cat_df["mean_activation"] + cat_df["std_activation"],
                    alpha=0.2, color="#2ca02c")
    ax.plot(cat_df["relative_t"], cat_df["mean_activation"], marker="o", markersize=5,
            linewidth=2, color="#2ca02c")
    ax.axvline(0, linestyle="--", color="red", linewidth=1.2, alpha=0.8, label="Violation onset")

    # Annotate the jump magnitude
    pre_mean = cat_df[cat_df["relative_t"] < 0]["mean_activation"].mean()
    post_mean = cat_df[cat_df["relative_t"] >= 0]["mean_activation"].mean()
    ratio = post_mean / max(pre_mean, 1e-8)
    ax.annotate(f"{ratio:.0f}x increase",
                xy=(1, post_mean), xytext=(3, post_mean * 1.5),
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))

    ax.set_xlabel("Timestep Relative to Violation Onset")
    ax.set_ylabel("Mean Top-Feature\nActivation")
    ax.set_title("Safety Features Activate at Boundary Violation Onset", pad=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_dual(fig, base_path)


def figure_per_suite(csv_path: Path, base_path: Path) -> None:
    """Figure 12: Per-suite violation rates and AUROC comparison."""
    if not csv_path.exists():
        save_placeholder(base_path, "Per-Suite Breakdown", "Run 08_additional_analyses.py first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        save_placeholder(base_path, "Per-Suite Breakdown", "No suite data found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=300)

    rate_cols = [c for c in df.columns if c.endswith("_rate") and c != "mean_top_k_activation"]
    rate_cols = [c for c in rate_cols if c != "auroc"]  # exclude auroc if somehow present
    label_map = {"collision_rate": "Collision", "excessive_force_rate": "Exc. Force",
                 "boundary_violation_rate": "Boundary", "high_approach_speed_rate": "High Speed",
                 "object_drop_rate": "Obj. Drop"}

    # Left panel: grouped bar chart of violation rates per suite
    suites = df["suite"].values
    x = np.arange(len(suites))
    n_cats = len(rate_cols)
    if n_cats > 0:
        width = 0.8 / n_cats
        cat_colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
        for i, col in enumerate(rate_cols):
            offset = (i - n_cats / 2 + 0.5) * width
            ax1.bar(x + offset, df[col].values, width=width,
                    label=label_map.get(col, col), color=cat_colors[i % len(cat_colors)])
        ax1.set_xticks(x)
        ax1.set_xticklabels(suites, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("Violation Rate")
        ax1.set_title("Violation Rates by Suite")
        ax1.legend(fontsize=6, loc="upper right")

    # Right panel: per-suite AUROC
    if "auroc" in df.columns:
        colors = sns.color_palette("colorblind", n_colors=len(suites))
        ax2.bar(x, df["auroc"].values, color=colors, width=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(suites, rotation=30, ha="right", fontsize=8)
        ax2.axhline(0.5, linestyle="--", color="gray", linewidth=1)
        ax2.set_ylabel("AUROC")
        ax2.set_title("Per-Suite Monitor AUROC")
        ax2.set_ylim(0.0, 1.0)

    fig.tight_layout()
    save_dual(fig, base_path)


def write_latex_tables(results_dir: Path, fig_dir: Path, paper_dir: Path, layer: int) -> None:
    monitor_dir = resolve_artifact_dir(results_dir, "monitor")
    causal_dir = resolve_artifact_dir(results_dir, "causal")
    differential_dir = resolve_artifact_dir(results_dir, "differential")
    ablation_dir = resolve_artifact_dir(results_dir, "ablations")

    monitor_csv = monitor_dir / f"layer{layer}_monitor_metrics.csv"
    cat_auroc_csv = monitor_dir / f"layer{layer}_per_category_auroc.csv"
    causal_csv = causal_dir / f"layer{layer}_causal_validation.csv"

    table_dir = paper_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: main methods (updated with model column).
    if monitor_csv.exists():
        mdf = pd.read_csv(monitor_csv)
        causal_baseline = np.nan
        causal_best = np.nan
        success_baseline = np.nan
        success_best = np.nan
        if causal_csv.exists():
            cdf = pd.read_csv(causal_csv)
            if "collision_rate_baseline" in cdf.columns:
                causal_baseline = float(cdf["collision_rate_baseline"].mean())
            if "collision_rate_clamped" in cdf.columns:
                causal_best = float(cdf["collision_rate_clamped"].mean())
            if "success_rate_baseline" in cdf.columns:
                success_baseline = float(cdf["success_rate_baseline"].mean())
            if "success_rate_clamped" in cdf.columns:
                success_best = float(cdf["success_rate_clamped"].mean())

        with (table_dir / "table_main_results.tex").open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{llcccc}\\toprule\n")
            f.write("Method & Model & AUROC$\\uparrow$ & F1$\\uparrow$ & Collision$\\downarrow$ & Success$\\uparrow$ \\\\ \\midrule\n")
            for _, row in mdf.iterrows():
                collision = "--"
                success = "--"
                if row["method"] == "sae_lr":
                    if not np.isnan(causal_baseline):
                        collision = f"{causal_baseline:.3f}"
                    if not np.isnan(success_baseline):
                        success = f"{success_baseline:.3f}"
                f.write(
                    f"{tex_escape(row['method'])} & OpenVLA & {row['auroc']:.3f} & {row['f1']:.3f} & {collision} & {success} \\\\ \n"
                )
            if not np.isnan(causal_best):
                success_s = "--" if np.isnan(success_best) else f"{success_best:.3f}"
                f.write(f"safesae+clamp(top-k) & OpenVLA & -- & -- & {causal_best:.3f} & {success_s} \\\\ \n")
            f.write("\\bottomrule\\end{tabular}\n")

    # Table 2: per-category stats + SAE LR AUROC.
    if cat_auroc_csv.exists():
        cdf = pd.read_csv(cat_auroc_csv)
        sae_rows = cdf[cdf["method"] == "sae_lr"] if "method" in cdf.columns else cdf

        def category_stats(category: str) -> tuple[str, str]:
            cat_csv = differential_dir / f"layer{layer}_{category}.csv"
            if not cat_csv.exists():
                cat_csv = differential_dir / f"openvla_layer{layer}_{category}.csv"
            if not cat_csv.exists():
                return "--", "--"
            cat_df = pd.read_csv(cat_csv)
            if cat_df.empty:
                return "0", "--"

            significant = "--"
            top_effect = "--"
            if "significant" in cat_df.columns:
                significant = str(int(cat_df["significant"].astype(bool).sum()))
            if "effect_size" in cat_df.columns:
                top_effect = f"{float(cat_df.iloc[0]['effect_size']):.3f}"
            return significant, top_effect
        with (table_dir / "table_category_auroc.tex").open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lccc}\\toprule\n")
            f.write("Category & Significant Features & Top Effect Size & AUROC$\\uparrow$ \\\\ \\midrule\n")
            for _, row in sae_rows.iterrows():
                sig_count, top_effect = category_stats(str(row["category"]))
                f.write(
                    f"{tex_escape(row['category'])} & {sig_count} & {top_effect} & {float(row['auroc']):.3f} \\\\ \n"
                )
            f.write("\\bottomrule\\end{tabular}\n")

    # Table 3: ablations.
    out_ablation = table_dir / "table_ablations.tex"
    if not ablation_dir.exists():
        with out_ablation.open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lcc}\\toprule\n")
            f.write("Variant & AUROC & Collision Rate \\\\ \\midrule\n")
            f.write("(populate from results/ablations/*.csv) & -- & -- \\\\ \n")
            f.write("\\bottomrule\\end{tabular}\n")
    else:
        rows = []
        for csv_path in sorted(ablation_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            name = csv_path.stem
            auroc_col = find_column(df, ["auroc", "roc_auc"]) or ""
            coll_col = find_column(df, ["collision_rate", "collision"]) or ""
            auroc = float(df[auroc_col].mean()) if auroc_col else np.nan
            coll = float(df[coll_col].mean()) if coll_col else np.nan
            rows.append((name, auroc, coll))

        with out_ablation.open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lcc}\\toprule\n")
            f.write("Variant & AUROC & Collision Rate \\\\ \\midrule\n")
            if rows:
                for name, auroc, coll in rows:
                    auroc_s = "--" if np.isnan(auroc) else f"{auroc:.3f}"
                    coll_s = "--" if np.isnan(coll) else f"{coll:.3f}"
                    f.write(f"{tex_escape(name)} & {auroc_s} & {coll_s} \\\\ \n")
            else:
                f.write("(no ablation rows found) & -- & -- \\\\ \n")
            f.write("\\bottomrule\\end{tabular}\n")

    # Table 4: dictionary size ablation (OpenVLA).
    out_table4 = table_dir / "table_dict_size_ablation.tex"
    d16 = differential_dir / "openvla_layer20_overall.csv"
    if not d16.exists():
        d16 = differential_dir / "layer20_overall.csv"
    d32 = differential_dir / "openvla_d32768_layer20_overall.csv"
    monitor_layer20 = monitor_dir / "layer20_monitor_metrics.csv"
    with out_table4.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\\toprule\n")
        f.write("Dict Size & Significant Features & Top-10 Mean Effect & AUROC & Dead Features \\% \\\\ \\midrule\n")

        def _row(df_path: Path) -> tuple[str, str, str]:
            if not df_path.exists():
                return "--", "--", "--"
            ddf = pd.read_csv(df_path)
            if ddf.empty:
                return "0", "0.000", "--"
            sig = str(int(ddf["significant"].astype(bool).sum())) if "significant" in ddf.columns else "--"
            eff = (
                f"{float(ddf['abs_effect_size'].head(10).mean()):.3f}"
                if "abs_effect_size" in ddf.columns
                else f"{float(np.abs(ddf['effect_size'].head(10)).mean()):.3f}"
            )
            auroc = "--"
            if monitor_layer20.exists():
                mdf = pd.read_csv(monitor_layer20)
                row = mdf[mdf["method"] == "sae_lr"]
                if not row.empty:
                    auroc = f"{float(row.iloc[0]['auroc']):.3f}"
            return sig, eff, auroc

        sig16, eff16, au16 = _row(d16)
        sig32, eff32, au32 = _row(d32)
        f.write(f"16K (4x) & {sig16} & {eff16} & {au16} & -- \\\\ \n")
        f.write(f"32K (8x) & {sig32} & {eff32} & {au32} & -- \\\\ \n")
        f.write("\\bottomrule\\end{tabular}\n")

    # Table 5: feature inspection (from additional analyses).
    additional_dir = results_dir / "additional"
    inspect_csv = additional_dir / "feature_inspection.csv"
    if inspect_csv.exists():
        idf = pd.read_csv(inspect_csv)
        with (table_dir / "table_feature_inspection.tex").open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{rllcccc}\\toprule\n")
            f.write("Rank & Feature & Top Category & Sparsity & Mean (viol) & Mean (safe) & Temporal \\\\ \\midrule\n")
            for _, row in idf.iterrows():
                f.write(
                    f"{int(row['rank'])} & {int(row['feature_idx'])} & {tex_escape(row['top_category'])} "
                    f"& {row['sparsity']:.3f} & {row['mean_viol']:.3f} & {row['mean_safe']:.3f} "
                    f"& {row['temporal_peak']} \\\\ \n"
                )
            f.write("\\bottomrule\\end{tabular}\n")

    # Copy key CSVs used by paper build.
    if monitor_csv.exists():
        pd.read_csv(monitor_csv).to_csv(fig_dir / f"layer{layer}_monitor_metrics.csv", index=False)
    if cat_auroc_csv.exists():
        pd.read_csv(cat_auroc_csv).to_csv(fig_dir / f"layer{layer}_per_category_auroc.csv", index=False)


def figure_auroc_bars(cat_auroc_csv: Path, base_path: Path) -> None:
    """Grouped bar chart comparing per-category AUROC across methods."""
    if not cat_auroc_csv.exists():
        save_placeholder(base_path, "AUROC Bar Chart", f"Missing {cat_auroc_csv.name}")
        return

    df = pd.read_csv(cat_auroc_csv)
    methods = ["sae_lr", "raw_activation_mlp", "force_threshold"]
    method_labels = {"sae_lr": "SAE LR", "raw_activation_mlp": "Raw MLP", "force_threshold": "Force Thresh."}
    method_colors = {"sae_lr": "#4C72B0", "raw_activation_mlp": "#55A868", "force_threshold": "#C44E52"}
    cat_labels = {
        "boundary_violation": "Boundary",
        "collision": "Collision",
        "excessive_force": "Exc. Force",
        "high_approach_speed": "High Speed",
        "object_drop": "Obj. Drop",
    }
    categories = list(cat_labels.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(categories))
    width = 0.25

    for i, method in enumerate(methods):
        mdf = df[df["method"] == method]
        vals = []
        for cat in categories:
            row = mdf[mdf["category"] == cat]
            vals.append(float(row["auroc"].iloc[0]) if len(row) > 0 else 0.0)
        ax.bar(x + i * width, vals, width, label=method_labels[method], color=method_colors[method], edgecolor="white", linewidth=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x + width)
    ax.set_xticklabels([cat_labels[c] for c in categories], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Per-Category Safety Detection AUROC by Method")
    fig.tight_layout()
    save_dual(fig, base_path)


def main() -> None:
    args = parse_args()
    res = Path(args.results_dir)
    out = Path(args.output_dir)
    paper_dir = Path(args.paper_dir)
    out.mkdir(parents=True, exist_ok=True)

    differential_dir = resolve_artifact_dir(res, "differential")
    monitor_dir = resolve_artifact_dir(res, "monitor")
    causal_dir = resolve_artifact_dir(res, "causal")
    ablation_dir = resolve_artifact_dir(res, "ablations")

    diff_csv = differential_dir / f"layer{args.layer}_overall.csv"
    if not diff_csv.exists():
        diff_csv = differential_dir / f"openvla_layer{args.layer}_overall.csv"
    overlap_csv = differential_dir / f"layer{args.layer}_category_overlap.csv"
    if not overlap_csv.exists():
        overlap_csv = differential_dir / f"openvla_layer{args.layer}_category_overlap.csv"
    monitor_csv = monitor_dir / f"layer{args.layer}_monitor_metrics.csv"
    roc_csv = monitor_dir / f"layer{args.layer}_roc_points.csv"
    per_cat_roc_csv = monitor_dir / f"layer{args.layer}_per_category_roc.csv"
    causal_csv = causal_dir / f"layer{args.layer}_causal_validation.csv"
    pareto_csv = monitor_dir / f"layer{args.layer}_pareto.csv"

    figure_architecture(out / "figure1_architecture")

    if diff_csv.exists():
        df = pd.read_csv(diff_csv)
        plot_volcano(df, str(out / "figure3_volcano.pdf"))
        plot_volcano(df, str(out / "figure3_volcano.png"))

    if overlap_csv.exists():
        overlap_df = pd.read_csv(overlap_csv)
        plot_overlap_heatmap(overlap_df, str(out / "figure6_overlap_heatmap.pdf"))
        plot_overlap_heatmap(overlap_df, str(out / "figure6_overlap_heatmap.png"))

    figure_roc_panels(roc_csv, per_cat_roc_csv, out / "figure4_roc")
    figure_causal_and_pareto(causal_csv, pareto_csv, out / "figure5_clamp_pareto")
    figure_ablations(ablation_dir, causal_csv, out / "figure6_ablations", results_dir=res)
    figure_cross_model_structure(res, out / "figure8_cross_model_structure")
    figure_cross_model_behavior(res, out / "figure9_cross_model_behavior")

    if args.data_dir and args.sae_checkpoint:
        figure_feature_heatmaps(
            data_dir=Path(args.data_dir),
            sae_checkpoint=Path(args.sae_checkpoint),
            sae_config_path=Path(args.sae_config),
            overall_csv=diff_csv,
            layer=args.layer,
            base_path=out / "figure2_feature_heatmaps",
        )
        figure_tsne(
            data_dir=Path(args.data_dir),
            sae_checkpoint=Path(args.sae_checkpoint),
            sae_config_path=Path(args.sae_config),
            layer=args.layer,
            max_episodes=args.max_tsne_episodes,
            base_path=out / "figure7_tsne",
        )

    # Additional analyses figures (Figures 10-12).
    additional_dir = res / "additional"
    figure_sparsity_curve(additional_dir / "sparsity_curve.csv", out / "figure10_sparsity_curve")
    figure_temporal_patterns(additional_dir / "temporal_patterns.csv", out / "figure11_temporal_patterns")
    figure_per_suite(additional_dir / "per_suite_breakdown.csv", out / "figure12_per_suite")

    # AUROC bar chart (per-category comparison across methods).
    cat_auroc_csv_fig = out / f"layer{args.layer}_per_category_auroc.csv"
    if not cat_auroc_csv_fig.exists():
        cat_auroc_csv_fig = monitor_dir / f"layer{args.layer}_per_category_auroc.csv"
    figure_auroc_bars(cat_auroc_csv_fig, out / "figure13_auroc_bars")

    write_latex_tables(res, out, paper_dir, args.layer)


if __name__ == "__main__":
    main()
