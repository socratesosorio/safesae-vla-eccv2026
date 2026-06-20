"""Generate SAE health and quick monitor AUROC table for workshop rebuttal checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Spec as label,path,layer,d_sae,k; e.g. layer20,/tmp/sae_layer20_d16384.pt,20,16384,32",
    )
    p.add_argument("--output_dir", type=str, default="logs/asap_workshop_experiments")
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_timesteps_per_episode", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_checkpoint_spec(spec: str) -> tuple[str, Path, int, int, int]:
    parts = [x.strip() for x in spec.split(",")]
    if len(parts) != 5:
        raise ValueError(f"Invalid --checkpoint spec: {spec}")
    label, path, layer, d_sae, k = parts
    return label, Path(path), int(layer), int(d_sae), int(k)


def sampled_indices(num_steps: int, max_steps: int) -> np.ndarray:
    if num_steps <= max_steps:
        return np.arange(num_steps, dtype=np.int64)
    return np.unique(np.linspace(0, num_steps - 1, num=max_steps, dtype=np.int64))


@torch.no_grad()
def collect_layer_samples(
    data_dir: Path,
    layer: int,
    max_episodes: int,
    max_timesteps_per_episode: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = f"activations_layer{layer}"
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    paths = sorted(data_dir.rglob("rollout_*.safetensors"))
    if max_episodes > 0:
        paths = paths[:max_episodes]
    for path in paths:
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            if key not in keys or "safety_labels" not in keys:
                continue
            acts = f.get_tensor(key).astype(np.float32)
            labels = f.get_tensor("safety_labels").astype(bool)
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        if labels.ndim == 3:
            labels = labels.any(axis=1)
        n = min(step_vecs.shape[0], labels.shape[0])
        if n == 0:
            continue
        idx = sampled_indices(n, max_timesteps_per_episode)
        xs.append(step_vecs[idx])
        ys.append(labels[idx].any(axis=1).astype(np.int64))
    if not xs:
        raise RuntimeError(f"No usable layer {layer} samples found under {data_dir}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


@torch.no_grad()
def encode_and_metrics(
    sae: torch.nn.Module,
    x: np.ndarray,
    norm_factor: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    x_t = torch.from_numpy(x).to(torch.float32) / float(max(norm_factor, 1e-8))
    chunks = []
    losses = []
    fvus = []
    l0s = []
    for start in range(0, x_t.shape[0], 1024):
        xb = x_t[start : start + 1024].to(device)
        _, metrics = sae.compute_loss(xb)
        feats = sae.encode(xb).detach().cpu().numpy().astype(np.float32)
        chunks.append(feats)
        losses.append(float(metrics["loss"]))
        fvus.append(float(metrics["fvu"]))
        l0s.append(float(metrics["l0"]))
    feats = np.concatenate(chunks, axis=0)
    active = int((feats > 0).any(axis=0).sum())
    return feats, {
        "active_features": active,
        "active_feature_pct": float(active / max(feats.shape[1], 1) * 100.0),
        "fvu": float(np.mean(fvus)),
        "l0": float(np.mean(l0s)),
        "loss": float(np.mean(losses)),
    }


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, score))


def quick_safety_auroc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    active = np.flatnonzero((features > 0).any(axis=0))
    if len(active) == 0:
        return 0.5
    x = features[:, active]
    idx = np.arange(x.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    if len(np.unique(labels[train_idx])) < 2:
        return 0.5
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=seed)
    clf.fit(x_train, labels[train_idx])
    return safe_auroc(labels[test_idx], clf.decision_function(x_test))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for spec in args.checkpoint:
        label, checkpoint_path, layer, d_sae, k = parse_checkpoint_spec(spec)
        if not checkpoint_path.exists():
            rows.append(
                {
                    "model": label,
                    "layer": layer,
                    "d_sae": d_sae,
                    "k": k,
                    "status": "missing_checkpoint",
                }
            )
            continue
        sae, norm_factor = load_sae_checkpoint(
            str(checkpoint_path),
            d_in=args.d_in,
            d_sae=d_sae,
            k=k,
            device=device,
        )
        x, y = collect_layer_samples(
            data_dir=Path(args.data_dir),
            layer=layer,
            max_episodes=args.max_episodes,
            max_timesteps_per_episode=args.max_timesteps_per_episode,
        )
        feats, metrics = encode_and_metrics(sae=sae, x=x, norm_factor=norm_factor, device=device)
        metrics["safety_auroc_quick_lr"] = quick_safety_auroc(feats, y, seed=args.seed)
        rows.append(
            {
                "model": label,
                "layer": layer,
                "d_sae": d_sae,
                "k": k,
                "norm_factor": float(norm_factor),
                "n_samples": int(x.shape[0]),
                "positive_rate": float(y.mean()),
                "status": "ok",
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sae_health_table.csv"
    df.to_csv(csv_path, index=False)
    tex_cols = [
        "model",
        "layer",
        "d_sae",
        "k",
        "active_features",
        "active_feature_pct",
        "fvu",
        "l0",
        "safety_auroc_quick_lr",
    ]
    df[[c for c in tex_cols if c in df.columns]].to_latex(
        out_dir / "table_sae_health.tex",
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.3f}",
    )
    with (out_dir / "sae_health_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"csv": str(csv_path), "num_rows": int(len(df))}, f, indent=2)


if __name__ == "__main__":
    main()
