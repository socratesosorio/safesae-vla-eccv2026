"""Progress differential analysis using SAE-encoded features (d_sae=16384).

The script:
1) Loads quartile-based episode labels (high/low progress).
2) Encodes sampled timestep activations with a provided SAE checkpoint.
3) Runs per-feature Mann-Whitney U tests + BH FDR correction.
4) Writes differential CSVs, top-k table, volcano, and heatmap artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from safetensors import safe_open
from scipy.stats import mannwhitneyu

from src.analysis.differential_activation import load_sae_checkpoint


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
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
    p = argparse.ArgumentParser(description="Progress differential analysis on SAE features")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing rollout_*.safetensors (recursive)")
    p.add_argument("--labels_csv", type=str, required=True, help="CSV with columns: episode_id,label (0/1)")
    p.add_argument("--labels_full_csv", type=str, default="", help="Optional full label CSV with progress_norm")
    p.add_argument("--sae_checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_timesteps_per_episode", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="logs/safesae_progress_sae_analysis")
    return p.parse_args()


def _sample_timestep_indices(num_steps: int, max_steps: int) -> np.ndarray:
    if num_steps <= max_steps:
        return np.arange(num_steps, dtype=np.int64)
    idx = np.linspace(0, num_steps - 1, num=max_steps, dtype=np.int64)
    return np.unique(idx)


@torch.no_grad()
def _encode_chunked(sae: torch.nn.Module, flat_acts: np.ndarray, norm_factor: float, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(flat_acts).to(torch.float32)
    x = x / float(max(norm_factor, 1e-8))
    outputs: list[torch.Tensor] = []
    chunk = 1024
    for i in range(0, x.shape[0], chunk):
        c = x[i : i + chunk].to(device)
        feats = sae.encode(c).detach().cpu()
        outputs.append(feats)
    return torch.cat(outputs, dim=0).numpy()


def collect_sae_samples(
    data_dir: Path,
    layer: int,
    label_map: dict[str, int],
    sae: torch.nn.Module,
    norm_factor: float,
    device: torch.device,
    max_timesteps_per_episode: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    key = f"activations_layer{layer}"
    low_samples: list[np.ndarray] = []
    high_samples: list[np.ndarray] = []
    episode_rows: list[dict[str, float | int | str]] = []

    for tensor_path in sorted(data_dir.rglob("rollout_*.safetensors")):
        episode_id = tensor_path.stem
        if episode_id not in label_map:
            continue
        label = int(label_map[episode_id])
        if label not in (0, 1):
            continue

        with safe_open(str(tensor_path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)  # [T, N, d_in]

        timestep_vecs = acts.mean(axis=1)  # [T, d_in], average tokens per step
        tids = _sample_timestep_indices(timestep_vecs.shape[0], max_timesteps_per_episode)
        sampled = timestep_vecs[tids]
        encoded = _encode_chunked(sae=sae, flat_acts=sampled, norm_factor=norm_factor, device=device)

        if label == 0:
            low_samples.append(encoded)
        else:
            high_samples.append(encoded)

        ep_mean = encoded.mean(axis=0)
        row: dict[str, float | int | str] = {"episode_id": episode_id, "label": label}
        for i, v in enumerate(ep_mean):
            row[f"f{i}"] = float(v)
        episode_rows.append(row)

    if not low_samples or not high_samples:
        raise RuntimeError("Insufficient labeled samples: one of low/high groups is empty.")
    low_matrix = np.concatenate(low_samples, axis=0)
    high_matrix = np.concatenate(high_samples, axis=0)
    return low_matrix, high_matrix, pd.DataFrame(episode_rows)


def run_diff(low_matrix: np.ndarray, high_matrix: np.ndarray, alpha: float) -> pd.DataFrame:
    n1, n2 = low_matrix.shape[0], high_matrix.shape[0]
    d_sae = low_matrix.shape[1]
    rows = []
    for feat_idx in range(d_sae):
        x = low_matrix[:, feat_idx]
        y = high_matrix[:, feat_idx]
        if float(np.max(x)) == 0.0 and float(np.max(y)) == 0.0:
            continue
        u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided")
        effect_size = 1.0 - (2.0 * float(u_stat)) / float(max(n1 * n2, 1))
        rows.append(
            {
                "feature_idx": int(feat_idx),
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
    ax.set_title("Progress Differential Volcano (SAE Features)")
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
    mat = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)
    sns.heatmap(mat, cmap="viridis", ax=ax, cbar_kws={"label": "z-score activation"})
    ax.set_xlabel("Episodes (ordered by progress)")
    ax.set_ylabel("Top differential SAE features")
    ax.set_yticks(np.arange(len(top_features)) + 0.5)
    ax.set_yticklabels([str(i) for i in top_features], fontsize=8)
    ax.set_title("Top SAE Features Across Episode Progress")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=args.sae_checkpoint,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        device=device,
    )

    low_matrix, high_matrix, episode_df = collect_sae_samples(
        data_dir=Path(args.data_dir),
        layer=args.layer,
        label_map=label_map,
        sae=sae,
        norm_factor=norm_factor,
        device=device,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
    )
    diff_df = run_diff(low_matrix=low_matrix, high_matrix=high_matrix, alpha=args.alpha)

    diff_csv = out_dir / f"layer{args.layer}_progress_overall_sae{args.d_sae}.csv"
    top20_csv = out_dir / f"top{args.top_k}_progress_features_sae{args.d_sae}.csv"
    ep_mean_csv = out_dir / f"episode_feature_means_sae{args.d_sae}.csv"
    diff_df.to_csv(diff_csv, index=False)
    diff_df.head(args.top_k).to_csv(top20_csv, index=False)
    episode_df.to_csv(ep_mean_csv, index=False)

    make_volcano(diff_df, out_dir / f"volcano_progress_sae{args.d_sae}")
    labels_full = pd.read_csv(args.labels_full_csv) if str(args.labels_full_csv).strip() else None
    top_feats = diff_df["feature_idx"].head(args.top_k).astype(int).tolist()
    make_progress_heatmap(
        episode_df=episode_df,
        top_features=top_feats,
        labels_full=labels_full,
        output_base=out_dir / f"heatmap_top_features_by_progress_sae{args.d_sae}",
    )

    summary = {
        "num_episodes_used": int(len(episode_df)),
        "num_low_progress_timestep_samples": int(low_matrix.shape[0]),
        "num_high_progress_timestep_samples": int(high_matrix.shape[0]),
        "num_features_tested": int(len(diff_df)),
        "num_significant_fdr_0_05": int(diff_df["significant"].sum()),
        "diff_csv": str(diff_csv),
        "topk_csv": str(top20_csv),
        "episode_means_csv": str(ep_mean_csv),
    }
    with (out_dir / f"summary_progress_sae{args.d_sae}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
