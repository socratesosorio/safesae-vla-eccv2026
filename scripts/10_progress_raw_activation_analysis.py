"""Fallback progress analysis using raw OpenVLA activations (4096-d).

This script is used when SAE checkpoints are unavailable.
It computes per-episode mean raw activations and performs a differential
Mann-Whitney U analysis between low-progress and high-progress episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from safetensors import safe_open
from scipy.stats import mannwhitneyu


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progress differential analysis on raw activations")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing rollout_*.safetensors (recursive)")
    p.add_argument("--labels_csv", type=str, required=True, help="CSV with columns: episode_id,label (0/1)")
    p.add_argument(
        "--labels_full_csv",
        type=str,
        default="",
        help="Optional full labels CSV with progress_norm for heatmap ordering",
    )
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="logs/safesae_progress_raw_analysis")
    return p.parse_args()


def load_episode_vectors(data_dir: Path, layer: int, label_map: dict[str, int]) -> pd.DataFrame:
    rows = []
    key = f"activations_layer{layer}"
    for tensor_path in sorted(data_dir.rglob("rollout_*.safetensors")):
        episode_id = tensor_path.stem
        if episode_id not in label_map:
            continue
        with safe_open(str(tensor_path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)  # [T, N, 4096]
        vec = acts.reshape(-1, acts.shape[-1]).mean(axis=0)  # [4096]
        row = {
            "episode_id": episode_id,
            "label": int(label_map[episode_id]),
        }
        for i, v in enumerate(vec):
            row[f"f{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def run_diff(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c.startswith("f")]
    low = df[df["label"] == 0]
    high = df[df["label"] == 1]
    n1, n2 = len(low), len(high)
    rows = []
    for col in feat_cols:
        x = low[col].to_numpy()
        y = high[col].to_numpy()
        if float(np.max(x)) == 0.0 and float(np.max(y)) == 0.0:
            continue
        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
        effect_size = 1.0 - (2.0 * float(u_stat)) / float(max(n1 * n2, 1))
        feat_idx = int(col[1:])
        rows.append(
            {
                "feature_idx": feat_idx,
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "effect_size": float(effect_size),
                "abs_effect_size": float(abs(effect_size)),
                "mean_low_progress": float(np.mean(x)),
                "mean_high_progress": float(np.mean(y)),
                "std_low_progress": float(np.std(x)),
                "std_high_progress": float(np.std(y)),
                "direction": "higher_in_high_progress" if effect_size < 0 else "higher_in_low_progress",
            }
        )
    out = pd.DataFrame(rows)
    out["adjusted_p"] = bh_fdr(out["p_value"].to_numpy())
    out["significant"] = out["adjusted_p"] < alpha
    out["composite_score"] = out["abs_effect_size"] * (-np.log10(out["adjusted_p"] + 1e-300))
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


def make_volcano(diff_df: pd.DataFrame, output_base: Path) -> None:
    sns.set_theme(style="whitegrid")
    x = diff_df["effect_size"].to_numpy()
    y = -np.log10(diff_df["adjusted_p"].to_numpy() + 1e-300)
    c = np.where(diff_df["significant"].to_numpy(), "tab:red", "tab:blue")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=250)
    ax.scatter(x, y, c=c, s=10, alpha=0.7)
    ax.set_xlabel("Effect Size (Rank-Biserial)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Progress Differential Volcano (Raw Activations)")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_progress_heatmap(
    episode_df: pd.DataFrame,
    top_features: list[int],
    labels_full: pd.DataFrame | None,
    output_base: Path,
) -> None:
    use = episode_df[["episode_id", "label"] + [f"f{i}" for i in top_features]].copy()
    if labels_full is not None and "progress_norm" in labels_full.columns:
        use = use.merge(labels_full[["episode_id", "progress_norm"]], on="episode_id", how="left")
        use = use.sort_values("progress_norm", ascending=True)
    else:
        use = use.sort_values("label", ascending=True)
    mat = use[[f"f{i}" for i in top_features]].to_numpy().T
    # Per-feature z-score for readability.
    mat = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)
    sns.heatmap(mat, cmap="viridis", ax=ax, cbar_kws={"label": "z-score activation"})
    ax.set_xlabel("Episodes (ordered by progress)")
    ax.set_ylabel("Top differential features")
    ax.set_yticks(np.arange(len(top_features)) + 0.5)
    ax.set_yticklabels([str(i) for i in top_features], fontsize=8)
    ax.set_title("Top Raw Features Across Episode Progress")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    label_map = {str(r["episode_id"]): int(r["label"]) for _, r in labels.iterrows()}
    episode_df = load_episode_vectors(Path(args.data_dir), args.layer, label_map)
    if episode_df.empty:
        raise RuntimeError("No labeled episodes found for analysis.")

    diff_df = run_diff(episode_df, alpha=args.alpha)
    diff_csv = out_dir / f"layer{args.layer}_progress_overall_raw4096.csv"
    diff_df.to_csv(diff_csv, index=False)

    top20_csv = out_dir / "top20_progress_features_raw4096.csv"
    diff_df.head(args.top_k).to_csv(top20_csv, index=False)

    make_volcano(diff_df, out_dir / "volcano_progress_raw4096")

    labels_full = pd.read_csv(args.labels_full_csv) if str(args.labels_full_csv).strip() else None
    top_feats = diff_df["feature_idx"].head(args.top_k).astype(int).tolist()
    make_progress_heatmap(
        episode_df=episode_df,
        top_features=top_feats,
        labels_full=labels_full,
        output_base=out_dir / "heatmap_top_features_by_progress_raw4096",
    )

    summary = {
        "num_episodes_used": int(len(episode_df)),
        "num_low_progress": int((episode_df["label"] == 0).sum()),
        "num_high_progress": int((episode_df["label"] == 1).sum()),
        "num_features_tested": int(len(diff_df)),
        "num_significant_fdr_0_05": int(diff_df["significant"].sum()),
        "top20_csv": str(top20_csv),
        "diff_csv": str(diff_csv),
    }
    with (out_dir / "summary_progress_raw4096.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
