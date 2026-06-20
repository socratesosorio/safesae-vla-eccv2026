"""Compare semantic-success baselines on rich success-labeled rollouts.

This is a gated rebuttal analysis: it should only be cited if the semantic
labels are non-degenerate and one of the inspectability-relevant SAE summaries
is competitive with motion/object-state controls. It uses cached OpenVLA
activations and does not run policy inference.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, required=True)
    p.add_argument("--sae_checkpoint", type=str, required=True)
    p.add_argument("--top_features_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_timesteps_per_episode", type=int, default=12)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _bootstrap_auroc_ci(y: np.ndarray, scores: np.ndarray, *, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float32)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_safe_auroc(y[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _read_meta(path: Path) -> dict:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _subsample_steps(step_vecs: np.ndarray, max_timesteps: int) -> np.ndarray:
    if step_vecs.shape[0] <= max_timesteps:
        return step_vecs
    idx = np.unique(np.linspace(0, step_vecs.shape[0] - 1, max_timesteps, dtype=np.int64))
    return step_vecs[idx]


@torch.no_grad()
def _load_episode_matrix(args: argparse.Namespace, top_features: list[int]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=args.sae_checkpoint,
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )
    act_key = f"activations_layer{int(args.layer)}"
    rows: list[dict] = []
    raw_means: list[np.ndarray] = []
    sae_means: list[np.ndarray] = []

    for path in sorted(Path(args.rollout_dir).rglob("*.safetensors")):
        meta = _read_meta(path)
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            if not {"rewards", "object_states", "success_flags", act_key, "eef_positions"} <= keys:
                continue
            acts = f.get_tensor(act_key).astype(np.float32)
            eef = f.get_tensor("eef_positions").astype(np.float32).reshape(-1, 3)
            rewards = f.get_tensor("rewards").astype(np.float32).reshape(-1)
            success_flags = f.get_tensor("success_flags").astype(bool).reshape(-1)
            object_states = f.get_tensor("object_states").astype(np.float32)
            success = bool(f.get_tensor("episode_success").astype(bool).any()) if "episode_success" in keys else bool(meta.get("episode_success", False))

        if eef.shape[0] < 2:
            continue
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        step_vecs = _subsample_steps(step_vecs, int(args.max_timesteps_per_episode))
        raw_mean = step_vecs.mean(axis=0).astype(np.float32, copy=False)
        x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
        sae_mean = sae.encode(x).detach().cpu().numpy().mean(axis=0).astype(np.float32, copy=False)

        eef_delta = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        object_motion = 0.0
        object_lift = False
        if object_states.ndim == 2 and object_states.shape[0] > 1 and object_states.shape[1] >= 3:
            object_motion = float(np.nanmax(np.linalg.norm(object_states[:, :3] - object_states[0, :3], axis=1)))
            object_lift = bool(np.nanmax(object_states[:, 2]) - float(object_states[0, 2]) > 0.04)
        reward_positive = bool(float(rewards.sum()) > 0 or (rewards.size and float(rewards.max()) > 0))

        rows.append(
            {
                "episode_id": path.stem,
                "suite": str(meta.get("suite", "unknown")),
                "success": int(success),
                "reward_positive": int(reward_positive),
                "object_lift": int(object_lift),
                "semantic_any": int(success or reward_positive or object_lift),
                "eef_displacement": float(np.linalg.norm(eef[-1] - eef[0])),
                "eef_path_length": float(eef_delta.sum()),
                "object_motion": object_motion,
                "any_success_flag": int(success_flags.any()),
            }
        )
        raw_means.append(raw_mean)
        sae_means.append(sae_mean)

    df = pd.DataFrame(rows)
    matrices = {
        "raw": np.vstack(raw_means).astype(np.float32) if raw_means else np.empty((0, int(args.d_in)), dtype=np.float32),
        "sae": np.vstack(sae_means).astype(np.float32) if sae_means else np.empty((0, int(args.d_sae)), dtype=np.float32),
    }
    matrices["top_sae"] = matrices["sae"][:, top_features] if len(df) else np.empty((0, len(top_features)), dtype=np.float32)
    return df, matrices


def _cv_scores(x: np.ndarray, y: np.ndarray, *, method: str, seed: int) -> dict:
    y = np.asarray(y, dtype=int)
    if len(y) < 8 or len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"auroc": float("nan"), "n": int(len(y)), "positives": int(y.sum()), "negatives": int((1 - y).sum())}
    n_splits = min(5, int(np.bincount(y).min()))
    yy, ss = [], []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y):
        if method.endswith("_pca20"):
            n_components = max(1, min(20, len(tr) - 1, x.shape[1]))
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=seed),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
            )
        model.fit(x[tr], y[tr])
        yy.append(y[te])
        ss.append(model.decision_function(x[te]))
    yy_arr = np.concatenate(yy)
    ss_arr = np.concatenate(ss)
    ci_low, ci_high = _bootstrap_auroc_ci(yy_arr, ss_arr, seed=seed)
    return {
        "auroc": _safe_auroc(yy_arr, ss_arr),
        "auroc_ci95_low": ci_low,
        "auroc_ci95_high": ci_high,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_features = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(int(args.top_k)).tolist()
    df, matrices = _load_episode_matrix(args, top_features)
    df.to_csv(out_dir / "success_labeled_episode_metadata.csv", index=False)

    geom = df[["eef_displacement", "eef_path_length"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 2), dtype=np.float32)
    geom_object = df[["eef_displacement", "eef_path_length", "object_motion"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 3), dtype=np.float32)
    specs = {
        "geometry_only": geom,
        "geometry_plus_object_motion": geom_object,
        "submitted_top20_sae": matrices["top_sae"],
        "submitted_top20_sae_plus_geometry": np.hstack([matrices["top_sae"], geom]) if len(df) else np.empty((0, 22), dtype=np.float32),
        "full_sae_lr": matrices["sae"],
        "full_sae_pca20": matrices["sae"],
        "raw_lr": matrices["raw"],
        "raw_pca20": matrices["raw"],
    }
    rows: list[dict] = []
    for target in ["success", "reward_positive", "object_lift", "semantic_any"]:
        if target not in df:
            continue
        y = df[target].to_numpy(dtype=int)
        for method, x in specs.items():
            res = _cv_scores(x, y, method=method, seed=int(args.seed))
            rows.append({"target": target, "method": method, **res})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "success_labeled_baseline_results.csv", index=False)
    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["success"].mean()) if len(df) else float("nan"),
        "reward_positive_rate": float(df["reward_positive"].mean()) if len(df) else float("nan"),
        "object_lift_rate": float(df["object_lift"].mean()) if len(df) else float("nan"),
        "semantic_any_rate": float(df["semantic_any"].mean()) if len(df) else float("nan"),
        "top_feature_nonzero_rate_mean": float((matrices["top_sae"] > 1e-8).mean()) if len(df) else float("nan"),
        "best_results": results.sort_values("auroc", ascending=False).head(16).to_dict(orient="records") if len(results) else [],
    }
    (out_dir / "success_labeled_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
