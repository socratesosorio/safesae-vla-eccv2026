"""Cross-model comparison utilities for OpenVLA vs pi0 safety feature analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.safety_labeler import SAFETY_CATEGORIES
from src.utils.runtime import ensure_dir


class CrossModelComparison:
    def __init__(self, openvla_results: dict[int, dict[str, pd.DataFrame]], pi0_results: dict[int, dict[str, pd.DataFrame]]):
        self.openvla = openvla_results
        self.pi0 = pi0_results

    @staticmethod
    def _layer_stats(results: dict[int, dict[str, pd.DataFrame]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for layer, rows in results.items():
            overall = rows.get("overall", pd.DataFrame())
            sig = int(overall["significant"].astype(bool).sum()) if not overall.empty and "significant" in overall else 0
            eff = float(overall["abs_effect_size"].head(50).mean()) if not overall.empty and "abs_effect_size" in overall else 0.0
            out[str(layer)] = {"significant": float(sig), "mean_effect_top50": eff}
        return out

    def structural_comparison(self) -> dict[str, Any]:
        openvla_stats = self._layer_stats(self.openvla)
        pi0_stats = self._layer_stats(self.pi0)

        cat_rows = []
        for category in SAFETY_CATEGORIES:
            ov_cat = []
            p_cat = []
            for rows in self.openvla.values():
                ov_cat.append(rows.get(category, pd.DataFrame()))
            for rows in self.pi0.values():
                p_cat.append(rows.get(category, pd.DataFrame()))

            ov_sig = int(sum(int(df["significant"].astype(bool).sum()) for df in ov_cat if not df.empty and "significant" in df))
            p_sig = int(sum(int(df["significant"].astype(bool).sum()) for df in p_cat if not df.empty and "significant" in df))
            ov_eff_vals = [float(df["abs_effect_size"].head(10).mean()) for df in ov_cat if not df.empty and "abs_effect_size" in df]
            p_eff_vals = [float(df["abs_effect_size"].head(10).mean()) for df in p_cat if not df.empty and "abs_effect_size" in df]
            ov_eff = float(np.mean(ov_eff_vals)) if ov_eff_vals else 0.0
            p_eff = float(np.mean(p_eff_vals)) if p_eff_vals else 0.0
            cat_rows.append(
                {
                    "category": category,
                    "openvla_count": ov_sig,
                    "pi0_count": p_sig,
                    "openvla_mean_effect": ov_eff,
                    "pi0_mean_effect": p_eff,
                }
            )

        return {
            "openvla_significant_by_layer": openvla_stats,
            "pi0_significant_by_layer": pi0_stats,
            "category_comparison": pd.DataFrame(cat_rows),
        }

    def behavioral_comparison(
        self,
        openvla_monitor_csv: str | None = None,
        pi0_monitor_csv: str | None = None,
        openvla_causal_json: str | None = None,
        pi0_causal_json: str | None = None,
    ) -> dict[str, Any]:
        def _overall_auroc(path: str | None) -> float:
            if not path or not Path(path).exists():
                return 0.0
            df = pd.read_csv(path)
            if df.empty:
                return 0.0
            row = df[df["method"] == "sae_lr"]
            if row.empty:
                return float(df.iloc[0]["auroc"])
            return float(row.iloc[0]["auroc"])

        def _collision(path: str | None) -> float:
            if not path or not Path(path).exists():
                return 0.0
            with Path(path).open("r", encoding="utf-8") as f:
                payload = json.load(f)
            clamped = payload.get("clamped", {})
            rate = clamped.get("collision_rate", {})
            if isinstance(rate, dict):
                return float(rate.get("mean", 0.0))
            return float(rate or 0.0)

        return {
            "openvla_auroc": _overall_auroc(openvla_monitor_csv),
            "pi0_auroc": _overall_auroc(pi0_monitor_csv),
            "openvla_collision_clamped": _collision(openvla_causal_json),
            "pi0_collision_clamped": _collision(pi0_causal_json),
        }

    def generate_comparison_figures(
        self,
        output_dir: str,
        openvla_monitor_csv: str | None = None,
        pi0_monitor_csv: str | None = None,
    ) -> None:
        out_dir = ensure_dir(output_dir)
        sns.set_theme(style="whitegrid", font_scale=1.1)

        structural = self.structural_comparison()
        ov = structural["openvla_significant_by_layer"]
        p0 = structural["pi0_significant_by_layer"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        ov_layers = sorted([int(k) for k in ov.keys()])
        p0_layers = sorted([int(k) for k in p0.keys()])
        x_labels = ["Early (~50%)", "Mid (~63%)", "Late (~75%)"]

        ov_counts = [int(ov.get(str(k), {}).get("significant", 0.0)) for k in ov_layers[:3]]
        p0_counts = [int(p0.get(str(k), {}).get("significant", 0.0)) for k in p0_layers[:3]]

        x = np.arange(len(x_labels))
        width = 0.36
        ax1.bar(x - width / 2, ov_counts, width=width, label="OpenVLA", color="#1f77b4")
        ax1.bar(x + width / 2, p0_counts, width=width, label="pi0", color="#ff7f0e")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels)
        ax1.set_ylabel("Significant features")
        ax1.set_title("Cross-model layer structure")
        ax1.legend()

        ov_effects = []
        p0_effects = []
        for rows in self.openvla.values():
            overall = rows.get("overall", pd.DataFrame())
            if not overall.empty and "abs_effect_size" in overall:
                ov_effects.extend(overall["abs_effect_size"].head(100).tolist())
        for rows in self.pi0.values():
            overall = rows.get("overall", pd.DataFrame())
            if not overall.empty and "abs_effect_size" in overall:
                p0_effects.extend(overall["abs_effect_size"].head(100).tolist())
        sns.violinplot(data=[ov_effects or [0.0], p0_effects or [0.0]], ax=ax2, palette=["#1f77b4", "#ff7f0e"])
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["OpenVLA", "pi0"])
        ax2.set_ylabel("|Effect size|")
        ax2.set_title("Top-feature effect distribution")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "fig8_cross_model_structure.pdf", bbox_inches="tight")
        fig.savefig(Path(out_dir) / "fig8_cross_model_structure.png", bbox_inches="tight")
        plt.close(fig)

        fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        auroc_rows = []
        if openvla_monitor_csv and Path(openvla_monitor_csv).exists():
            ov_df = pd.read_csv(openvla_monitor_csv)
            for _, row in ov_df.iterrows():
                auroc_rows.append({"method": row.get("method", "unknown"), "model": "OpenVLA", "auroc": row.get("auroc", 0.0)})
        if pi0_monitor_csv and Path(pi0_monitor_csv).exists():
            p0_df = pd.read_csv(pi0_monitor_csv)
            for _, row in p0_df.iterrows():
                auroc_rows.append({"method": row.get("method", "unknown"), "model": "pi0", "auroc": row.get("auroc", 0.0)})
        if auroc_rows:
            adf = pd.DataFrame(auroc_rows)
            target = adf[adf["method"] == "sae_lr"] if "sae_lr" in set(adf["method"]) else adf
            sns.barplot(data=target, x="model", y="auroc", ax=bx1, palette=["#1f77b4", "#ff7f0e"])
        bx1.set_ylim(0.0, 1.0)
        bx1.axhline(0.5, linestyle="--", color="gray", linewidth=1)
        bx1.set_title("Overall AUROC")

        comp = self.structural_comparison()["category_comparison"]
        if not comp.empty:
            cat_df = comp.melt(
                id_vars=["category"],
                value_vars=["openvla_count", "pi0_count"],
                var_name="model",
                value_name="value",
            )
            sns.barplot(data=cat_df, x="category", y="value", hue="model", ax=bx2)
            bx2.tick_params(axis="x", rotation=20)
            bx2.set_title("Category significant feature counts")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "fig9_cross_model_behavior.pdf", bbox_inches="tight")
        fig.savefig(Path(out_dir) / "fig9_cross_model_behavior.png", bbox_inches="tight")
        plt.close(fig)
