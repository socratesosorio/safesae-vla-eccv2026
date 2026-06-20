"""Generate early-prefix and telemetry-phase checks for ECCV rebuttal.

This is a secondary add-on after the confound controls. It avoids semantic
claims that require object state, using only cached activations, EEF telemetry,
actions, contact force, and safety labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="data/rollouts")
    p.add_argument(
        "--labels_full_csv",
        type=str,
        default="logs/safesae_progress_labels/progress_labels_full.csv",
    )
    p.add_argument(
        "--top_features_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv",
    )
    p.add_argument("--sae_checkpoint", type=str, default="data/sae_checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_temporal_phase_checks")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--prefixes", type=str, default="0.10,0.25,0.50,0.75,1.0")
    p.add_argument("--max_timesteps_per_prefix", type=int, default=16)
    p.add_argument("--max_phase_steps_per_episode", type=int, default=64)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def metric_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    vals = []
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(np.asarray(vals), [2.5, 97.5]))


def make_global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    return df.reset_index(drop=True)


def fold_iterator(y: np.ndarray, *, folds: int, seed: int):
    y = np.asarray(y, dtype=int)
    n_splits = min(int(folds), int(np.bincount(y).min())) if len(np.unique(y)) == 2 else 0
    if n_splits < 2:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=int(seed),
            stratify=y if len(np.unique(y)) == 2 else None,
        )
        yield train_idx, test_idx
        return
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    yield from splitter.split(np.zeros(len(y)), y)


def fit_predict_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
            scaler = StandardScaler(with_mean=False)
            x_train = scaler.fit_transform(x[train_idx]).astype(np.float32, copy=False)
            x_test = scaler.transform(x[test_idx]).astype(np.float32, copy=False)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                solver="liblinear",
                random_state=int(seed),
            )
            clf.fit(x_train, y[train_idx])
            scores[test_idx] = clf.decision_function(x_test)
    return y, scores


def summarize_cv(
    *,
    prefix: float,
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    yy, scores = fit_predict_cv(x, y, seed=seed, folds=folds)
    auroc_ci = metric_ci(yy, scores, metric_fn=safe_auroc, rng=rng, n_boot=bootstrap)
    pr_ci = metric_ci(yy, scores, metric_fn=safe_pr_auc, rng=rng, n_boot=bootstrap)
    return {
        "prefix_fraction": float(prefix),
        "method": method,
        "n": int(len(yy)),
        "positives": int(yy.sum()),
        "negatives": int((1 - yy).sum()),
        "auroc": safe_auroc(yy, scores),
        "auroc_ci95_low": auroc_ci[0],
        "auroc_ci95_high": auroc_ci[1],
        "pr_auc": safe_pr_auc(yy, scores),
        "pr_auc_ci95_low": pr_ci[0],
        "pr_auc_ci95_high": pr_ci[1],
    }


def parse_prefixes(spec: str) -> list[float]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val <= 0 or val > 1:
            raise ValueError(f"prefix must be in (0,1], got {val}")
        out.append(val)
    return sorted(set(out))


def select_indices(n: int, max_items: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_items <= 0 or n <= max_items:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_items, dtype=np.int64))


def motion_features(
    *,
    eef: np.ndarray,
    actions: np.ndarray,
    contact: np.ndarray | None,
    safety: np.ndarray | None,
    n: int,
) -> list[float]:
    eef = np.asarray(eef, dtype=np.float32).reshape(-1, 3)[:n]
    actions = np.asarray(actions, dtype=np.float32).reshape(actions.shape[0], -1)[:n]
    contact_arr = np.asarray(contact, dtype=np.float32)[:n] if contact is not None else np.zeros((n, 1), dtype=np.float32)
    safety_arr = np.asarray(safety, dtype=bool)[:n] if safety is not None else np.zeros((n, 1), dtype=bool)
    eef_delta = np.linalg.norm(np.diff(eef, axis=0), axis=1) if len(eef) > 1 else np.zeros((0,), dtype=np.float32)
    action_norm = np.linalg.norm(actions, axis=1) if len(actions) else np.zeros((0,), dtype=np.float32)
    trans_norm = np.linalg.norm(actions[:, :3], axis=1) if actions.shape[1] >= 3 else action_norm
    gripper = actions[:, 6] if actions.shape[1] > 6 else np.zeros((len(actions),), dtype=np.float32)
    contact_norm = np.linalg.norm(contact_arr.reshape(len(contact_arr), -1), axis=1) if len(contact_arr) else np.zeros((0,), dtype=np.float32)
    return [
        float(len(eef)),
        float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) > 1 else 0.0,
        float(eef_delta.sum()) if len(eef_delta) else 0.0,
        float(eef_delta.mean()) if len(eef_delta) else 0.0,
        float(eef_delta.max()) if len(eef_delta) else 0.0,
        float(action_norm.mean()) if len(action_norm) else 0.0,
        float(action_norm.max()) if len(action_norm) else 0.0,
        float(trans_norm.mean()) if len(trans_norm) else 0.0,
        float(gripper.mean()) if len(gripper) else 0.0,
        float(np.abs(gripper).mean()) if len(gripper) else 0.0,
        float(contact_norm.mean()) if len(contact_norm) else 0.0,
        float(contact_norm.max()) if len(contact_norm) else 0.0,
        float(safety_arr.any(axis=1).mean()) if safety_arr.ndim == 2 and len(safety_arr) else 0.0,
    ]


MOTION_COLS = [
    "num_steps_prefix",
    "eef_final_displacement_prefix",
    "eef_path_length_prefix",
    "eef_mean_velocity_proxy_prefix",
    "eef_max_velocity_proxy_prefix",
    "action_mean_norm_prefix",
    "action_max_norm_prefix",
    "action_translation_mean_norm_prefix",
    "gripper_action_mean_prefix",
    "gripper_action_mean_abs_prefix",
    "contact_force_mean_prefix",
    "contact_force_max_prefix",
    "safety_active_fraction_prefix",
]


@torch.no_grad()
def encode_steps(
    sae,
    norm_factor: float,
    step_vecs: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
    return sae.encode(x).detach().cpu().numpy().astype(np.float32, copy=False)


def phase_masks(
    *,
    eef: np.ndarray,
    actions: np.ndarray,
    contact: np.ndarray | None,
    safety: np.ndarray | None,
    idx: np.ndarray,
) -> dict[str, np.ndarray]:
    t = len(eef)
    if t == 0:
        return {}
    norm_t = np.arange(t) / max(t - 1, 1)
    speed = np.zeros((t,), dtype=np.float32)
    if t > 1:
        speed[1:] = np.linalg.norm(np.diff(eef.reshape(-1, 3), axis=0), axis=1)
    speed_hi = np.quantile(speed, 0.75) if len(speed) else 0.0
    speed_lo = np.quantile(speed, 0.25) if len(speed) else 0.0
    actions = actions.reshape(actions.shape[0], -1)
    gripper = actions[:, 6] if actions.shape[1] > 6 else np.zeros((t,), dtype=np.float32)
    contact_norm = np.zeros((t,), dtype=np.float32)
    if contact is not None:
        c = np.asarray(contact, dtype=np.float32).reshape(t, -1)
        contact_norm = np.linalg.norm(c, axis=1)
    contact_hi = np.quantile(contact_norm, 0.75) if len(contact_norm) else 0.0
    safety_any = np.asarray(safety, dtype=bool).any(axis=1) if safety is not None and np.asarray(safety).ndim == 2 else np.zeros((t,), dtype=bool)
    onset_window = np.zeros((t,), dtype=bool)
    if safety_any.any():
        first = int(np.argmax(safety_any))
        onset_window[max(0, first - 2) : min(t, first + 3)] = True
    masks = {
        "early_third": norm_t < 1 / 3,
        "middle_third": (norm_t >= 1 / 3) & (norm_t < 2 / 3),
        "late_third": norm_t >= 2 / 3,
        "eef_speed_high": speed >= speed_hi,
        "eef_speed_low_stall_proxy": speed <= speed_lo,
        "gripper_action_positive": gripper > 0,
        "gripper_action_negative": gripper < 0,
        "gripper_action_high_abs": np.abs(gripper) >= np.quantile(np.abs(gripper), 0.75),
        "contact_force_high": contact_norm >= contact_hi,
        "safety_active": safety_any,
        "safety_onset_window": onset_window,
    }
    return {name: mask[idx].astype(bool) for name, mask in masks.items()}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    labels_full = pd.read_csv(args.labels_full_csv)
    labels = make_global_quartile_labels(labels_full)
    if int(args.max_episodes) > 0:
        labels = labels.head(int(args.max_episodes)).copy()
    y = labels["label"].to_numpy(dtype=int)
    top_features = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(20).tolist()
    prefixes = parse_prefixes(args.prefixes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=str(args.sae_checkpoint),
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )

    rollout_map = {p.stem: p for p in Path(args.rollout_dir).rglob("rollout_*.safetensors")}
    by_prefix: dict[float, dict[str, list]] = {
        prefix: {"episode_id": [], "y": [], "full": [], "motion": []} for prefix in prefixes
    }
    phase_values: dict[tuple[str, int], dict[str, list[float]]] = {}

    key = f"activations_layer{int(args.layer)}"
    used_episodes = 0
    for _, row in labels.iterrows():
        ep = str(row["episode_id"])
        path = rollout_map.get(ep)
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)
            eef = f.get_tensor("eef_positions").astype(np.float32)
            actions = f.get_tensor("actions").astype(np.float32)
            contact = f.get_tensor("contact_forces").astype(np.float32) if "contact_forces" in f.keys() else None
            safety = f.get_tensor("safety_labels").astype(bool) if "safety_labels" in f.keys() else None

        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        t = int(step_vecs.shape[0])
        if t == 0:
            continue
        used_episodes += 1
        for prefix in prefixes:
            n = max(1, int(math.ceil(t * float(prefix))))
            idx = select_indices(n, int(args.max_timesteps_per_prefix))
            feats = encode_steps(sae, norm_factor, step_vecs[idx], device=device).mean(axis=0)
            by_prefix[prefix]["episode_id"].append(ep)
            by_prefix[prefix]["y"].append(int(row["label"]))
            by_prefix[prefix]["full"].append(feats)
            by_prefix[prefix]["motion"].append(
                motion_features(eef=eef, actions=actions, contact=contact, safety=safety, n=n)
            )

        phase_idx = select_indices(t, int(args.max_phase_steps_per_episode))
        phase_feats = encode_steps(sae, norm_factor, step_vecs[phase_idx], device=device)[:, top_features]
        masks = phase_masks(eef=eef.reshape(-1, 3), actions=actions, contact=contact, safety=safety, idx=phase_idx)
        for phase_name, mask in masks.items():
            other = ~mask
            for j, feat_idx in enumerate(top_features):
                key_tuple = (phase_name, int(feat_idx))
                bucket = phase_values.setdefault(key_tuple, {"phase": [], "other": []})
                if mask.any():
                    bucket["phase"].extend(phase_feats[mask, j].astype(float).tolist())
                if other.any():
                    bucket["other"].extend(phase_feats[other, j].astype(float).tolist())

    pred_rows = []
    for prefix in prefixes:
        pack = by_prefix[prefix]
        yy = np.asarray(pack["y"], dtype=int)
        if len(yy) == 0:
            continue
        full = np.vstack(pack["full"]).astype(np.float32, copy=False)
        motion = np.vstack(pack["motion"]).astype(np.float32, copy=False)
        top = full[:, top_features]
        pred_rows.append(
            summarize_cv(
                prefix=prefix,
                method="top20_sae",
                x=top,
                y=yy,
                seed=seed,
                folds=int(args.folds),
                bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        pred_rows.append(
            summarize_cv(
                prefix=prefix,
                method="full_sae",
                x=full,
                y=yy,
                seed=seed,
                folds=int(args.folds),
                bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        pred_rows.append(
            summarize_cv(
                prefix=prefix,
                method="motion_telemetry",
                x=motion,
                y=yy,
                seed=seed,
                folds=int(args.folds),
                bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
        pred_rows.append(
            summarize_cv(
                prefix=prefix,
                method="motion_plus_top20_sae",
                x=np.concatenate([motion, top], axis=1),
                y=yy,
                seed=seed,
                folds=int(args.folds),
                bootstrap=int(args.bootstrap),
                rng=rng,
            )
        )
    prefix_df = pd.DataFrame(pred_rows)
    prefix_df.to_csv(out_dir / "early_prefix_prediction.csv", index=False)

    phase_rows = []
    for (phase_name, feat_idx), vals in sorted(phase_values.items()):
        phase_vals = np.asarray(vals["phase"], dtype=float)
        other_vals = np.asarray(vals["other"], dtype=float)
        mean_phase = float(np.mean(phase_vals)) if len(phase_vals) else float("nan")
        mean_other = float(np.mean(other_vals)) if len(other_vals) else float("nan")
        phase_rows.append(
            {
                "phase": phase_name,
                "feature_idx": int(feat_idx),
                "n_phase_steps": int(len(phase_vals)),
                "n_other_steps": int(len(other_vals)),
                "mean_activation_phase": mean_phase,
                "mean_activation_other": mean_other,
                "delta_phase_minus_other": float(mean_phase - mean_other)
                if np.isfinite(mean_phase) and np.isfinite(mean_other)
                else float("nan"),
                "ratio_phase_over_other": float(mean_phase / max(mean_other, 1e-8))
                if np.isfinite(mean_phase) and np.isfinite(mean_other)
                else float("nan"),
            }
        )
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(out_dir / "phase_feature_enrichment.csv", index=False)

    summary = {
        "seed": seed,
        "rollout_dir": str(args.rollout_dir),
        "sae_checkpoint": str(args.sae_checkpoint),
        "n_labeled_episodes": int(len(labels)),
        "n_used_episodes": int(used_episodes),
        "prefixes": prefixes,
        "top_features": top_features,
        "motion_columns": MOTION_COLS,
        "best_prefix_rows": prefix_df.sort_values("auroc", ascending=False).head(10).to_dict(orient="records")
        if len(prefix_df)
        else [],
        "largest_phase_enrichments": phase_df.reindex(
            phase_df["delta_phase_minus_other"].abs().sort_values(ascending=False).index
        )
        .head(20)
        .to_dict(orient="records")
        if len(phase_df)
        else [],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
