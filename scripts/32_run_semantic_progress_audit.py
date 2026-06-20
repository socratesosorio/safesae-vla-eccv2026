"""Evaluate whether SAE progress features predict semantic task signals.

This consumes newly collected rich LIBERO rollouts that include rewards,
success flags, and object-state tensors. It compares top SAE features against
geometric displacement controls and reports whether sparse progress features
predict semantic success/stage labels beyond movement-only information.
"""

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, required=True)
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--sae_checkpoint", type=str, default="results/athena_pilot/checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/semantic_progress_audit")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_timesteps_per_episode", type=int, default=12)
    p.add_argument("--require_semantic_keys", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _fit_cv(x: np.ndarray, y: np.ndarray, *, seed: int) -> dict:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"auroc": float("nan"), "n": int(len(y)), "positives": int(y.sum()), "negatives": int((1 - y).sum())}
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits >= 2:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
        splits = [(tr, te)]
    yy, ss = [], []
    for tr, te in splits:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
        )
        model.fit(x[tr], y[tr])
        scores = model.decision_function(x[te])
        yy.append(y[te])
        ss.append(scores)
    yy_arr = np.concatenate(yy)
    ss_arr = np.concatenate(ss)
    return {
        "auroc": _safe_auroc(yy_arr, ss_arr),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }


def _read_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@torch.no_grad()
def _encode_episode_features(
    *,
    sae,
    norm_factor: float,
    device: torch.device,
    acts: np.ndarray,
    max_timesteps: int,
) -> np.ndarray:
    step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
    if step_vecs.shape[0] > max_timesteps:
        idx = np.unique(np.linspace(0, step_vecs.shape[0] - 1, max_timesteps, dtype=np.int64))
        step_vecs = step_vecs[idx]
    x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
    return sae.encode(x).detach().cpu().numpy().mean(axis=0)


def _episode_rows(args: argparse.Namespace, top_features: list[int]) -> tuple[pd.DataFrame, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=args.sae_checkpoint,
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )
    key = f"activations_layer{int(args.layer)}"
    rows = []
    available_key_sets: list[str] = []
    for path in sorted(Path(args.rollout_dir).rglob("*.safetensors")):
        meta = _read_meta(path)
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            available_key_sets.append(" ".join(sorted(keys)))
            if args.require_semantic_keys and not {"rewards", "object_states", "success_flags"} <= keys:
                continue
            if key not in keys or "eef_positions" not in keys:
                continue
            acts = f.get_tensor(key).astype(np.float32)
            eef = f.get_tensor("eef_positions").astype(np.float32).reshape(-1, 3)
            rewards = f.get_tensor("rewards").astype(np.float32).reshape(-1) if "rewards" in keys else np.zeros((0,), dtype=np.float32)
            success = bool(f.get_tensor("episode_success").astype(bool).any()) if "episode_success" in keys else bool(meta.get("episode_success", False))
            success_flags = f.get_tensor("success_flags").astype(bool).reshape(-1) if "success_flags" in keys else np.zeros((0,), dtype=bool)
            object_states = f.get_tensor("object_states").astype(np.float32) if "object_states" in keys else np.zeros((0, 0), dtype=np.float32)

        if eef.shape[0] < 2:
            continue
        feats = _encode_episode_features(
            sae=sae,
            norm_factor=norm_factor,
            device=device,
            acts=acts,
            max_timesteps=int(args.max_timesteps_per_episode),
        )
        eef_delta = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        displacement = float(np.linalg.norm(eef[-1] - eef[0]))
        path_length = float(eef_delta.sum())
        reward_sum = float(rewards.sum()) if rewards.size else float(meta.get("reward_sum", 0.0) or 0.0)
        reward_positive = bool(reward_sum > 0 or (rewards.size and float(rewards.max()) > 0))
        first_success_stage = float(np.argmax(success_flags) / max(len(success_flags) - 1, 1)) if success_flags.any() else float("nan")

        object_lift = False
        object_motion = 0.0
        if object_states.ndim == 2 and object_states.shape[0] > 1 and object_states.shape[1] >= 3:
            z = object_states[:, 2]
            object_motion = float(np.nanmax(np.linalg.norm(object_states[:, :3] - object_states[0, :3], axis=1)))
            object_lift = bool(np.nanmax(z) - float(z[0]) > 0.04)

        row = {
            "episode_id": path.stem,
            "suite": str(meta.get("suite", "unknown")),
            "success": int(success),
            "reward_positive": int(reward_positive),
            "object_lift": int(object_lift),
            "semantic_any": int(success or reward_positive or object_lift),
            "first_success_stage": first_success_stage,
            "reward_sum": reward_sum,
            "eef_displacement": displacement,
            "eef_path_length": path_length,
            "object_motion": object_motion,
        }
        row.update({f"f{i}": float(feats[i]) for i in top_features if i < feats.shape[0]})
        rows.append(row)
    return pd.DataFrame(rows), sorted(set(available_key_sets))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_features = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(int(args.top_k)).tolist()
    df, key_sets = _episode_rows(args, top_features)
    df.to_csv(out_dir / "semantic_progress_episode_features.csv", index=False)

    top_cols = [f"f{i}" for i in top_features if f"f{i}" in df.columns]
    geom_cols = ["eef_displacement", "eef_path_length"]
    results = []
    for target in ["success", "reward_positive", "object_lift", "semantic_any"]:
        if target not in df.columns or df.empty:
            continue
        y = df[target].to_numpy(dtype=int)
        specs = [
            ("geometry_only", geom_cols),
            ("top_sae", top_cols),
            ("top_sae_plus_geometry", top_cols + geom_cols),
        ]
        if "object_motion" in df.columns:
            specs.append(("geometry_plus_object_motion", geom_cols + ["object_motion"]))
        for method, cols in specs:
            if not cols:
                continue
            res = _fit_cv(df[cols].to_numpy(dtype=np.float32), y, seed=int(args.seed))
            results.append({"target": target, "method": method, **res})
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "semantic_progress_audit_results.csv", index=False)

    summary = {
        "n_episodes": int(len(df)),
        "available_key_sets": key_sets[:10],
        "success_rate": float(df["success"].mean()) if "success" in df and len(df) else float("nan"),
        "reward_positive_rate": float(df["reward_positive"].mean()) if "reward_positive" in df and len(df) else float("nan"),
        "object_lift_rate": float(df["object_lift"].mean()) if "object_lift" in df and len(df) else float("nan"),
        "semantic_any_rate": float(df["semantic_any"].mean()) if "semantic_any" in df and len(df) else float("nan"),
        "top_feature_nonzero_rate_mean": float((df[top_cols].to_numpy(dtype=np.float32) > 1e-8).mean()) if top_cols and len(df) else float("nan"),
        "top_feature_variance_max": float(df[top_cols].var().max()) if top_cols and len(df) else float("nan"),
        "best_results": res_df.sort_values("auroc", ascending=False).head(10).to_dict(orient="records") if not res_df.empty else [],
    }
    (out_dir / "semantic_progress_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
