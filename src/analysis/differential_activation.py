"""Statistical identification of safety-correlated SAE features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from src.data.activation_dataset import AnalysisDataset
from src.data.safety_labeler import SAFETY_CATEGORIES
from src.sae.model import BatchTopKSAE
from src.sae.train_sae import BatchTopKSAE as LegacyBatchTopKSAE
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir


class DifferentialActivationAnalyzer:
    def __init__(self, sae: BatchTopKSAE, config: dict, norm_factor: float = 1.0):
        self.sae = sae
        self.config = config
        self.device = next(self.sae.parameters()).device
        self.norm_factor = float(max(norm_factor, 1e-8))

    @torch.no_grad()
    def compute_episode_feature_vector(self, episode_data: dict, layer: int) -> np.ndarray:
        key = f"activations_layer{int(layer)}"
        if key not in episode_data:
            raise KeyError(f"Missing key {key} in episode payload")

        activations = episode_data[key]  # [T, N, d_in] where N=7 for OpenVLA, 1 for pi0
        flat = activations.reshape(-1, activations.shape[-1]).to(torch.float32) / self.norm_factor

        chunk_size = 1024
        all_feats: list[torch.Tensor] = []
        for i in range(0, flat.shape[0], chunk_size):
            chunk = flat[i : i + chunk_size].to(self.device)
            feats = self.sae.encode(chunk)
            all_feats.append(feats.detach().cpu())
        merged = torch.cat(all_feats, dim=0)
        return merged.mean(dim=0).numpy()

    def _run_single_analysis(self, safe_features: np.ndarray, unsafe_features: np.ndarray, alpha: float) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        d_sae = int(safe_features.shape[1])
        n1 = int(safe_features.shape[0])
        n2 = int(unsafe_features.shape[0])

        for feat_idx in tqdm(range(d_sae), desc="Per-feature tests"):
            safe_vals = safe_features[:, feat_idx]
            unsafe_vals = unsafe_features[:, feat_idx]

            if float(safe_vals.max()) == 0.0 and float(unsafe_vals.max()) == 0.0:
                continue

            try:
                u_stat, p_val = mannwhitneyu(safe_vals, unsafe_vals, alternative="two-sided")
            except ValueError:
                continue

            effect_size = 1.0 - (2.0 * float(u_stat)) / float(max(n1 * n2, 1))
            rows.append(
                {
                    "feature_idx": int(feat_idx),
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                    "effect_size": float(effect_size),
                    "abs_effect_size": float(abs(effect_size)),
                    "mean_safe": float(np.mean(safe_vals)),
                    "mean_unsafe": float(np.mean(unsafe_vals)),
                    "std_safe": float(np.std(safe_vals)),
                    "std_unsafe": float(np.std(unsafe_vals)),
                    "freq_safe": float(np.mean(safe_vals > 0)),
                    "freq_unsafe": float(np.mean(unsafe_vals > 0)),
                    "direction": "higher_in_unsafe" if effect_size < 0 else "higher_in_safe",
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        reject, adj_p, _, _ = multipletests(df["p_value"], alpha=alpha, method="fdr_bh")
        df["adjusted_p"] = adj_p
        df["significant"] = reject
        df["composite_score"] = df["abs_effect_size"] * (-np.log10(df["adjusted_p"] + 1e-300))
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def run_layer_analysis(
        self,
        dataset: AnalysisDataset,
        layer: int,
        episode_labels: dict[str, int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        analysis_cfg = self.config.get("analysis", self.config.get("safety_analysis", {}))
        alpha = float(analysis_cfg.get("significance_level", 0.05))

        per_episode_features: dict[int, np.ndarray] = {}
        for idx in tqdm(range(len(dataset)), desc=f"Compute episode SAE features (layer {layer})"):
            per_episode_features[idx] = self.compute_episode_feature_vector(dataset[idx], layer=layer)

        # Explicit progress split mode: use externally provided episode labels.
        if episode_labels is not None:
            low_rows: list[np.ndarray] = []
            high_rows: list[np.ndarray] = []
            for ep_idx in range(len(dataset)):
                episode_id = Path(dataset.files[ep_idx]).stem
                label = episode_labels.get(episode_id, None)
                if label is None:
                    continue
                if int(label) == 1:
                    high_rows.append(per_episode_features[ep_idx])
                elif int(label) == 0:
                    low_rows.append(per_episode_features[ep_idx])
            if not low_rows or not high_rows:
                empty = pd.DataFrame(
                    columns=[
                        "feature_idx",
                        "u_statistic",
                        "p_value",
                        "effect_size",
                        "abs_effect_size",
                        "mean_safe",
                        "mean_unsafe",
                        "std_safe",
                        "std_unsafe",
                        "freq_safe",
                        "freq_unsafe",
                        "direction",
                        "adjusted_p",
                        "significant",
                        "composite_score",
                    ]
                )
                return {"overall": empty}
            low_matrix = np.stack(low_rows)
            high_matrix = np.stack(high_rows)
            # Keep legacy column names for downstream compatibility:
            # "safe" -> low_progress, "unsafe" -> high_progress.
            return {"overall": self._run_single_analysis(low_matrix, high_matrix, alpha=alpha)}

        categories = ["overall"] + SAFETY_CATEGORIES
        # When all episodes have violations, fall back to severity-based split:
        # bottom quartile (fewest violations) = "safe", top quartile = "unsafe".
        severity_quantile = float(analysis_cfg.get("severity_quantile", 0.25))
        out: dict[str, pd.DataFrame] = {}
        for category in categories:
            safe_rows: list[np.ndarray] = []
            unsafe_rows: list[np.ndarray] = []

            # First try binary classification.
            binary_safe = []
            binary_unsafe = []
            for ep_idx in range(len(dataset)):
                meta = dataset.metadata[ep_idx]
                if category == "overall":
                    is_unsafe = bool(meta.get("has_violations", False))
                else:
                    counts = meta.get("violation_counts", {}) or {}
                    is_unsafe = int(counts.get(category, 0)) > 0
                if is_unsafe:
                    binary_unsafe.append(ep_idx)
                else:
                    binary_safe.append(ep_idx)

            if binary_safe and binary_unsafe:
                # Normal case: both groups exist.
                safe_rows = [per_episode_features[i] for i in binary_safe]
                unsafe_rows = [per_episode_features[i] for i in binary_unsafe]
            else:
                # Severity fallback: split by violation count quartiles.
                violation_counts = []
                for ep_idx in range(len(dataset)):
                    meta = dataset.metadata[ep_idx]
                    if category == "overall":
                        count = int(meta.get("total_violations", 0))
                    else:
                        counts = meta.get("violation_counts", {}) or {}
                        count = int(counts.get(category, 0))
                    violation_counts.append((ep_idx, count))
                violation_counts.sort(key=lambda x: x[1])
                n = len(violation_counts)
                low_cutoff = int(n * severity_quantile)
                high_cutoff = int(n * (1.0 - severity_quantile))
                if low_cutoff > 0 and high_cutoff < n:
                    safe_rows = [per_episode_features[idx] for idx, _ in violation_counts[:low_cutoff]]
                    unsafe_rows = [per_episode_features[idx] for idx, _ in violation_counts[high_cutoff:]]

            if not safe_rows or not unsafe_rows:
                out[category] = pd.DataFrame(
                    columns=[
                        "feature_idx",
                        "u_statistic",
                        "p_value",
                        "effect_size",
                        "abs_effect_size",
                        "mean_safe",
                        "mean_unsafe",
                        "std_safe",
                        "std_unsafe",
                        "freq_safe",
                        "freq_unsafe",
                        "direction",
                        "adjusted_p",
                        "significant",
                        "composite_score",
                    ]
                )
                continue

            safe_matrix = np.stack(safe_rows)
            unsafe_matrix = np.stack(unsafe_rows)
            out[category] = self._run_single_analysis(safe_matrix, unsafe_matrix, alpha=alpha)

        return out

    @staticmethod
    def compare_layers(results_by_layer: dict[int, dict[str, pd.DataFrame]], top_k: int = 50) -> dict[str, Any]:
        out: dict[str, Any] = {
            "num_significant_features_per_layer": {},
            "mean_effect_size_per_layer": {},
            "feature_overlap_jaccard": {},
        }

        top_sets: dict[int, set[int]] = {}
        for layer, layer_results in results_by_layer.items():
            overall = layer_results.get("overall", pd.DataFrame())
            if overall.empty:
                out["num_significant_features_per_layer"][str(layer)] = 0
                out["mean_effect_size_per_layer"][str(layer)] = 0.0
                top_sets[layer] = set()
                continue
            sig = int(overall["significant"].astype(bool).sum()) if "significant" in overall.columns else 0
            mean_eff = float(overall["abs_effect_size"].head(top_k).mean()) if "abs_effect_size" in overall.columns else 0.0
            out["num_significant_features_per_layer"][str(layer)] = sig
            out["mean_effect_size_per_layer"][str(layer)] = mean_eff
            top_sets[layer] = set(overall["feature_idx"].head(top_k).astype(int).tolist())

        layers = sorted(results_by_layer.keys())
        for i, la in enumerate(layers):
            for lb in layers[i + 1 :]:
                a = top_sets.get(la, set())
                b = top_sets.get(lb, set())
                union = len(a | b)
                jacc = float(len(a & b) / union) if union else 0.0
                out["feature_overlap_jaccard"][f"{la}_{lb}"] = jacc
        return out


def load_sae_checkpoint(path: str, d_in: int, d_sae: int, k: int, device: torch.device) -> tuple[BatchTopKSAE, float]:
    ckpt = torch.load(path, map_location=device)
    d_in = int(ckpt.get("d_in", d_in))
    d_sae = int(ckpt.get("d_sae", d_sae))
    k = int(ckpt.get("k", k))
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    last_exc: Exception | None = None
    model = None
    for cls in (BatchTopKSAE, LegacyBatchTopKSAE):
        try:
            model = cls(d_in=d_in, d_sae=d_sae, k=k).to(device)  # type: ignore[call-arg]
            model.load_state_dict(state)
            model.eval()
            break
        except Exception as exc:
            last_exc = exc
            model = None
    if model is None:
        raise RuntimeError(f"Unable to load SAE checkpoint {path}: {last_exc}")
    norm_factor = float(ckpt.get("norm_factor", 1.0))
    return model, norm_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run differential activation analysis")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--output_dir", type=str, default="results/analysis/differential")
    parser.add_argument(
        "--progress_labels_csv",
        type=str,
        default="",
        help="Optional CSV with columns: episode_id,label where label=1 high_progress, 0 low_progress",
    )
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sae_cfg = load_yaml(args.sae_config)
    eval_cfg = load_yaml(args.eval_config)
    sae_primary = sae_cfg.get("primary", sae_cfg.get("sae", sae_cfg))
    d_in = int(sae_primary.get("d_in", 4096))
    d_sae = int(sae_primary.get("d_sae", 16384))
    k = int(sae_primary.get("k", 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(args.sae_checkpoint, d_in=d_in, d_sae=d_sae, k=k, device=device)
    analyzer = DifferentialActivationAnalyzer(sae=sae, config=eval_cfg, norm_factor=norm_factor)
    dataset = AnalysisDataset(data_dir=args.data_dir, test_split=args.test_split, seed=args.seed)
    episode_labels: dict[str, int] | None = None
    if str(args.progress_labels_csv).strip():
        labels_df = pd.read_csv(args.progress_labels_csv)
        if "episode_id" not in labels_df.columns or "label" not in labels_df.columns:
            raise ValueError("progress_labels_csv must contain columns: episode_id,label")
        episode_labels = {
            str(row["episode_id"]): int(row["label"])
            for _, row in labels_df.iterrows()
        }
    layer_results = analyzer.run_layer_analysis(dataset, layer=args.layer, episode_labels=episode_labels)

    out_dir = ensure_dir(args.output_dir)
    for name, df in layer_results.items():
        df.to_csv(Path(out_dir) / f"layer{args.layer}_{name}.csv", index=False)

    summary = {
        "layer": int(args.layer),
        "overall_significant": int(layer_results["overall"]["significant"].astype(bool).sum())
        if not layer_results["overall"].empty and "significant" in layer_results["overall"].columns
        else 0,
    }
    with (Path(out_dir) / f"layer{args.layer}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
