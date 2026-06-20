"""Generate cached safety-burden checks for the ECCV rebuttal.

The submitted 750-rollout cache has degenerate episode_success values and dense
safety-label counts. This script therefore avoids success/safe-vs-unsafe claims
and instead asks whether SAE features diagnose high-vs-low safety burden.

The main target is constructed within each LIBERO suite: top quartile of final
safety-label burden vs. bottom quartile. Early-prefix rows use only prefix
activations/telemetry to predict final safety burden.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint


SAFETY_CATEGORIES = [
    "collision",
    "excessive_force",
    "boundary_violation",
    "high_approach_speed",
    "object_drop",
]

MOTION_NO_CONTACT_COLS = [
    "eef_final_displacement",
    "eef_path_length",
    "eef_mean_velocity_proxy",
    "eef_max_velocity_proxy",
    "action_mean_norm",
    "action_max_norm",
    "action_translation_mean_norm",
]

MOTION_WITH_CONTACT_COLS = MOTION_NO_CONTACT_COLS + [
    "contact_force_mean",
    "contact_force_max",
]

PREFIX_MOTION_NO_CONTACT_COLS = [
    "prefix_steps",
    "prefix_eef_displacement",
    "prefix_eef_path_length",
    "prefix_eef_mean_velocity",
    "prefix_eef_max_velocity",
    "prefix_action_mean_norm",
    "prefix_action_max_norm",
    "prefix_action_translation_mean_norm",
    "prefix_gripper_mean",
    "prefix_gripper_abs_mean",
]

PREFIX_MOTION_WITH_CONTACT_COLS = PREFIX_MOTION_NO_CONTACT_COLS + [
    "prefix_contact_mean",
    "prefix_contact_max",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="data/rollouts")
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--sae_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_sae_layer20_d16384_means_all.csv")
    p.add_argument("--raw_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_raw_layer20_means.csv")
    p.add_argument(
        "--telemetry_csv",
        type=str,
        default="logs/eccv_rebuttal_checks/episode_telemetry_controls_and_semantic_audit.csv",
    )
    p.add_argument(
        "--top_features_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv",
    )
    p.add_argument("--sae_checkpoint", type=str, default="data/sae_checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_safety_burden_checks")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--prefixes", type=str, default="0.10,0.25")
    p.add_argument("--max_timesteps_per_prefix", type=int, default=16)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--skip_prefix", action="store_true")
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def parse_prefixes(spec: str) -> list[float]:
    values = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0 or value > 1:
            raise ValueError(f"prefix must be in (0, 1], got {value}")
        values.append(value)
    return sorted(set(values))


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def metric_ci(y: np.ndarray, scores: np.ndarray, metric_fn, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals: list[float] = []
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    y = y[valid]
    scores = scores[valid]
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        val = metric_fn(y[idx], scores[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


def paired_delta_ci(
    y: np.ndarray,
    base_scores: np.ndarray,
    add_scores: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    base_scores = np.asarray(base_scores, dtype=float)
    add_scores = np.asarray(add_scores, dtype=float)
    valid = np.isfinite(base_scores) & np.isfinite(add_scores)
    y = y[valid]
    base_scores = base_scores[valid]
    add_scores = add_scores[valid]
    observed = metric_fn(y, add_scores) - metric_fn(y, base_scores)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(metric_fn(y[idx], add_scores[idx]) - metric_fn(y[idx], base_scores[idx])))
    if not vals:
        return float(observed), float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(observed), float(lo), float(hi)


def fold_iterator(y: np.ndarray, folds: int, seed: int):
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


def fit_lr_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=np.float32)
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
    return scores


def fit_pca20_lr_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=np.float32)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
            scaler = StandardScaler(with_mean=True)
            x_train = scaler.fit_transform(x[train_idx]).astype(np.float32, copy=False)
            x_test = scaler.transform(x[test_idx]).astype(np.float32, copy=False)
            n_components = min(20, x_train.shape[0] - 1, x_train.shape[1])
            pca = PCA(n_components=max(1, n_components), random_state=int(seed))
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                solver="liblinear",
                random_state=int(seed),
            )
            clf.fit(x_train_pca, y[train_idx])
            scores[test_idx] = clf.decision_function(x_test_pca)
    return scores


def fit_ridge_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=np.float32)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x[train_idx]).astype(np.float32, copy=False)
        x_test = scaler.transform(x[test_idx]).astype(np.float32, copy=False)
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=int(seed))
        clf.fit(x_train, y[train_idx])
        scores[test_idx] = clf.decision_function(x_test)
    return scores


def fit_nested_top20_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> tuple[np.ndarray, list[list[int]]]:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=np.float32)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    selected: list[list[int]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
            x_train = x[train_idx]
            y_train = y[train_idx].astype(np.float32)
            centered_y = y_train - y_train.mean()
            centered_x = x_train - x_train.mean(axis=0, keepdims=True)
            denom = (x_train.std(axis=0) + 1e-8) * (y_train.std() + 1e-8)
            corr = (centered_x * centered_y[:, None]).mean(axis=0) / denom
            top_idx = np.argsort(-np.abs(corr))[:20]
            selected.append([int(i) for i in top_idx])
            scaler = StandardScaler(with_mean=False)
            x_tr = scaler.fit_transform(x[train_idx][:, top_idx]).astype(np.float32, copy=False)
            x_te = scaler.transform(x[test_idx][:, top_idx]).astype(np.float32, copy=False)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                solver="liblinear",
                random_state=int(seed),
            )
            clf.fit(x_tr, y[train_idx])
            scores[test_idx] = clf.decision_function(x_te)
    return scores, selected


def summarize_scores(
    *,
    target: str,
    method: str,
    y: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    bootstrap: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auroc_ci = metric_ci(y, scores, safe_auroc, rng, bootstrap)
    pr_ci = metric_ci(y, scores, safe_pr_auc, rng, bootstrap)
    row: dict[str, Any] = {
        "target": target,
        "method": method,
        "n": int(len(y)),
        "positives": int(np.asarray(y, dtype=int).sum()),
        "negatives": int((1 - np.asarray(y, dtype=int)).sum()),
        "auroc": safe_auroc(y, scores),
        "auroc_ci95_low": auroc_ci[0],
        "auroc_ci95_high": auroc_ci[1],
        "pr_auc": safe_pr_auc(y, scores),
        "pr_auc_ci95_low": pr_ci[0],
        "pr_auc_ci95_high": pr_ci[1],
    }
    if extra:
        row.update(extra)
    return row


def within_suite_quartile_labels(stats: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for suite, part in stats.groupby("suite", sort=True):
        values = part[target_col].astype(float)
        q25, q75 = values.quantile([0.25, 0.75])
        if not np.isfinite(q25) or not np.isfinite(q75) or q25 >= q75:
            continue
        low = part[values <= q25].copy()
        high = part[values >= q75].copy()
        low["burden_label"] = 0
        high["burden_label"] = 1
        low["burden_target"] = target_col
        high["burden_target"] = target_col
        low["burden_q25"] = float(q25)
        high["burden_q25"] = float(q25)
        low["burden_q75"] = float(q75)
        high["burden_q75"] = float(q75)
        rows.extend([low, high])
    if not rows:
        return pd.DataFrame(columns=list(stats.columns) + ["burden_label", "burden_target", "burden_q25", "burden_q75"])
    return pd.concat(rows, ignore_index=True)


def load_safety_stats(rollout_dir: Path, labels_full: pd.DataFrame, max_episodes: int = 0) -> pd.DataFrame:
    labels = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    labels["episode_id"] = labels["episode_id"].astype(str)
    if int(max_episodes) > 0:
        labels = labels.head(int(max_episodes)).copy()
    rollout_map = {p.stem: p for p in rollout_dir.rglob("rollout_*.safetensors")}
    rows = []
    for _, row in labels.iterrows():
        ep = str(row["episode_id"])
        path = rollout_map.get(ep)
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            if "safety_labels" not in keys:
                continue
            safety = f.get_tensor("safety_labels").astype(np.float32)
            if safety.ndim == 1:
                safety = safety[:, None]
            t = int(max(1, safety.shape[0]))
            cat_counts = safety.sum(axis=0)
            payload: dict[str, Any] = {
                "episode_id": ep,
                "suite": str(row["suite"]),
                "progress_norm": float(row["progress_norm"]),
                "num_steps": t,
                "total_safety_count": float(safety.sum()),
                "total_burden_rate": float(safety.sum() / t),
                "any_safety_active_fraction": float((safety.sum(axis=1) > 0).mean()),
            }
            if "episode_success" in keys:
                payload["episode_success"] = float(np.asarray(f.get_tensor("episode_success")).mean())
            for i, name in enumerate(SAFETY_CATEGORIES):
                count = float(cat_counts[i]) if i < len(cat_counts) else 0.0
                payload[f"{name}_count"] = count
                payload[f"{name}_rate"] = float(count / t)
            rows.append(payload)
    return pd.DataFrame(rows)


def load_design(args: argparse.Namespace, target: pd.DataFrame) -> pd.DataFrame:
    sae = pd.read_csv(args.sae_features_csv)
    raw = pd.read_csv(args.raw_features_csv)
    tel = pd.read_csv(args.telemetry_csv).drop(columns=["suite"], errors="ignore")
    for frame in [sae, raw, tel, target]:
        frame["episode_id"] = frame["episode_id"].astype(str)
    df = target.merge(sae, on="episode_id").merge(raw, on="episode_id").merge(tel, on="episode_id")
    return df


def get_top20_cols(path: str, df: pd.DataFrame) -> list[str]:
    top = pd.read_csv(path)
    cols = [f"f{int(x)}" for x in top["feature_idx"].head(20).tolist()]
    return [c for c in cols if c in df.columns]


def evaluate_target(
    *,
    df: pd.DataFrame,
    target_name: str,
    top20_cols: list[str],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["burden_label"].to_numpy(dtype=int)
    fcols = feature_cols(df)
    rcols = raw_cols(df)
    rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    def add_prediction(method: str, scores: np.ndarray, extra: dict[str, Any] | None = None) -> None:
        rows.append(
            summarize_scores(
                target=target_name,
                method=method,
                y=y,
                scores=scores,
                rng=rng,
                bootstrap=int(args.bootstrap),
                extra=extra,
            )
        )
        pred_rows.append(pd.DataFrame({"episode_id": df["episode_id"], "target": target_name, "method": method, "y": y, "score": scores}))

    suite = pd.get_dummies(df["suite"].astype(str), prefix="suite", dtype=float).to_numpy(np.float32)
    add_prediction("suite_id_only", fit_lr_cv(suite, y, seed=int(args.seed), folds=int(args.folds)))

    if top20_cols:
        top20 = df[top20_cols].to_numpy(np.float32)
        add_prediction("submitted_top20_sae", fit_lr_cv(top20, y, seed=int(args.seed), folds=int(args.folds)))

    full_sae = df[fcols].to_numpy(np.float32)
    nested_scores, selected = fit_nested_top20_cv(full_sae, y, seed=int(args.seed), folds=int(args.folds))
    selected_ids = [[int(fcols[i][1:]) for i in fold] for fold in selected]
    overlap = np.zeros((len(selected), len(selected)), dtype=float)
    for i, a in enumerate(selected_ids):
        for j, b in enumerate(selected_ids):
            overlap[i, j] = len(set(a).intersection(b))
    mean_overlap = float(overlap[np.triu_indices_from(overlap, k=1)].mean()) if len(selected) > 1 else float("nan")
    add_prediction(
        "nested_train_top20_sae",
        nested_scores,
        extra={"mean_pairwise_top20_overlap": mean_overlap, "fold_selected_features": json.dumps(selected_ids)},
    )
    add_prediction("full_sae_ridge", fit_ridge_cv(full_sae, y, seed=int(args.seed), folds=int(args.folds)))

    if rcols:
        raw = df[rcols].to_numpy(np.float32)
        add_prediction("raw_pca20_lr", fit_pca20_lr_cv(raw, y, seed=int(args.seed), folds=int(args.folds)))

    motion_no_contact_cols = [c for c in MOTION_NO_CONTACT_COLS if c in df.columns]
    motion_with_contact_cols = [c for c in MOTION_WITH_CONTACT_COLS if c in df.columns]
    if motion_no_contact_cols:
        motion_no = df[motion_no_contact_cols].to_numpy(np.float32)
        add_prediction("motion_no_contact", fit_lr_cv(motion_no, y, seed=int(args.seed), folds=int(args.folds)))
        if top20_cols:
            add_prediction(
                "motion_no_contact_plus_submitted_top20",
                fit_lr_cv(np.concatenate([motion_no, df[top20_cols].to_numpy(np.float32)], axis=1), y, seed=int(args.seed), folds=int(args.folds)),
            )
    if motion_with_contact_cols:
        motion_with = df[motion_with_contact_cols].to_numpy(np.float32)
        add_prediction("motion_with_contact_upper_bound", fit_lr_cv(motion_with, y, seed=int(args.seed), folds=int(args.folds)))
        if top20_cols:
            add_prediction(
                "motion_with_contact_plus_submitted_top20",
                fit_lr_cv(np.concatenate([motion_with, df[top20_cols].to_numpy(np.float32)], axis=1), y, seed=int(args.seed), folds=int(args.folds)),
            )

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    deltas = []
    if not predictions.empty:
        for base, add in [
            ("motion_no_contact", "motion_no_contact_plus_submitted_top20"),
            ("motion_with_contact_upper_bound", "motion_with_contact_plus_submitted_top20"),
        ]:
            base_df = predictions[predictions["method"] == base].sort_values("episode_id")
            add_df = predictions[predictions["method"] == add].sort_values("episode_id")
            if len(base_df) and len(add_df) and list(base_df["episode_id"]) == list(add_df["episode_id"]):
                dy = base_df["y"].to_numpy(dtype=int)
                for metric_name, metric_fn in [("auroc", safe_auroc), ("pr_auc", safe_pr_auc)]:
                    obs, lo, hi = paired_delta_ci(
                        dy,
                        base_df["score"].to_numpy(float),
                        add_df["score"].to_numpy(float),
                        metric_fn,
                        rng,
                        int(args.bootstrap),
                    )
                    deltas.append(
                        {
                            "target": target_name,
                            "base_method": base,
                            "added_method": add,
                            "metric": metric_name,
                            "delta": obs,
                            "delta_ci95_low": lo,
                            "delta_ci95_high": hi,
                        }
                    )
    return pd.DataFrame(rows), predictions, pd.DataFrame(deltas)


def run_fdr(df: pd.DataFrame, target_name: str, alpha: float) -> pd.DataFrame:
    fcols = feature_cols(df)
    low = df[df["burden_label"] == 0][fcols].to_numpy(np.float32)
    high = df[df["burden_label"] == 1][fcols].to_numpy(np.float32)
    rows = []
    for j, col in enumerate(fcols):
        low_vals = low[:, j]
        high_vals = high[:, j]
        if float(low_vals.max()) == 0.0 and float(high_vals.max()) == 0.0:
            continue
        try:
            stat, p_val = mannwhitneyu(low_vals, high_vals, alternative="two-sided")
        except ValueError:
            continue
        n_low = max(1, len(low_vals))
        n_high = max(1, len(high_vals))
        effect = 1.0 - (2.0 * float(stat)) / float(n_low * n_high)
        rows.append(
            {
                "target": target_name,
                "feature_idx": int(col[1:]),
                "u_statistic": float(stat),
                "p_value": float(p_val),
                "effect_size": float(effect),
                "abs_effect_size": float(abs(effect)),
                "mean_low_burden": float(low_vals.mean()),
                "mean_high_burden": float(high_vals.mean()),
                "freq_low_burden": float((low_vals > 0).mean()),
                "freq_high_burden": float((high_vals > 0).mean()),
                "direction": "higher_in_high_burden" if float(high_vals.mean()) >= float(low_vals.mean()) else "higher_in_low_burden",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    reject, adj_p, _, _ = multipletests(out["p_value"], alpha=float(alpha), method="fdr_bh")
    out["adjusted_p"] = adj_p
    out["significant"] = reject
    out["composite_score"] = out["abs_effect_size"] * (-np.log10(out["adjusted_p"] + 1e-300))
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


def select_indices(n: int, max_items: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_items <= 0 or n <= max_items:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_items, dtype=np.int64))


@torch.no_grad()
def encode_steps(sae, norm_factor: float, step_vecs: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
    return sae.encode(x).detach().cpu().numpy().astype(np.float32, copy=False)


def prefix_motion_features(eef: np.ndarray, actions: np.ndarray, contact: np.ndarray | None, n: int) -> list[float]:
    eef = np.asarray(eef, dtype=np.float32).reshape(-1, 3)[:n]
    actions = np.asarray(actions, dtype=np.float32).reshape(actions.shape[0], -1)[:n]
    contact_arr = np.asarray(contact, dtype=np.float32)[:n] if contact is not None else np.zeros((n, 1), dtype=np.float32)
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
    ]


def run_prefix_checks(
    *,
    labels: pd.DataFrame,
    top_feature_ids: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefixes = parse_prefixes(args.prefixes)
    rollout_map = {p.stem: p for p in Path(args.rollout_dir).rglob("rollout_*.safetensors")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=str(args.sae_checkpoint),
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )
    key = f"activations_layer{int(args.layer)}"
    by_prefix: dict[float, dict[str, list[Any]]] = {p: {"episode_id": [], "y": [], "top20": [], "motion": []} for p in prefixes}
    for _, row in labels.iterrows():
        ep = str(row["episode_id"])
        path = rollout_map.get(ep)
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            if key not in keys:
                continue
            acts = f.get_tensor(key).astype(np.float32)
            eef = f.get_tensor("eef_positions").astype(np.float32)
            actions = f.get_tensor("actions").astype(np.float32)
            contact = f.get_tensor("contact_forces").astype(np.float32) if "contact_forces" in keys else None
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        t = int(step_vecs.shape[0])
        if t <= 0:
            continue
        for prefix in prefixes:
            n = max(1, int(math.ceil(t * float(prefix))))
            idx = select_indices(n, int(args.max_timesteps_per_prefix))
            feats = encode_steps(sae, norm_factor, step_vecs[idx], device=device).mean(axis=0)
            by_prefix[prefix]["episode_id"].append(ep)
            by_prefix[prefix]["y"].append(int(row["burden_label"]))
            by_prefix[prefix]["top20"].append(feats[top_feature_ids])
            by_prefix[prefix]["motion"].append(prefix_motion_features(eef, actions, contact, n))

    rows = []
    pred_rows = []
    for prefix, pack in by_prefix.items():
        y = np.asarray(pack["y"], dtype=int)
        if len(y) == 0:
            continue
        top20 = np.vstack(pack["top20"]).astype(np.float32, copy=False)
        motion = np.vstack(pack["motion"]).astype(np.float32, copy=False)
        motion_no = motion[:, : len(PREFIX_MOTION_NO_CONTACT_COLS)]
        method_x = {
            "prefix_submitted_top20_sae": top20,
            "prefix_motion_no_contact": motion_no,
            "prefix_motion_with_contact_upper_bound": motion,
            "prefix_motion_no_contact_plus_submitted_top20": np.concatenate([motion_no, top20], axis=1),
            "prefix_motion_with_contact_plus_submitted_top20": np.concatenate([motion, top20], axis=1),
        }
        for method, x in method_x.items():
            scores = fit_lr_cv(x, y, seed=int(args.seed), folds=int(args.folds))
            row = summarize_scores(
                target=f"final_total_burden_rate_within_suite_q25_q75_prefix_{prefix:g}",
                method=method,
                y=y,
                scores=scores,
                rng=rng,
                bootstrap=int(args.bootstrap),
                extra={"prefix_fraction": float(prefix)},
            )
            rows.append(row)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "episode_id": pack["episode_id"],
                        "prefix_fraction": float(prefix),
                        "method": method,
                        "y": y,
                        "score": scores,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    labels_full = pd.read_csv(args.labels_full_csv)
    labels_full["episode_id"] = labels_full["episode_id"].astype(str)
    safety_stats = load_safety_stats(Path(args.rollout_dir), labels_full, max_episodes=int(args.max_episodes))
    if safety_stats.empty:
        raise RuntimeError(f"No safety labels found under {args.rollout_dir}")
    safety_stats.to_csv(out_dir / "safety_burden_episode_stats.csv", index=False)

    audit_rows = []
    target_specs = ["total_burden_rate"] + [f"{name}_rate" for name in SAFETY_CATEGORIES]
    all_results = []
    all_predictions = []
    all_deltas = []
    all_fdr = []

    for target_col in target_specs:
        target_labels = within_suite_quartile_labels(safety_stats, target_col)
        if target_labels.empty or target_labels["burden_label"].nunique() < 2:
            audit_rows.append({"target": target_col, "usable": False, "n": 0, "positives": 0, "negatives": 0})
            continue
        counts = target_labels.groupby(["suite", "burden_label"]).size().unstack(fill_value=0)
        audit_rows.append(
            {
                "target": target_col,
                "usable": True,
                "n": int(len(target_labels)),
                "positives": int(target_labels["burden_label"].sum()),
                "negatives": int((1 - target_labels["burden_label"]).sum()),
                "suite_label_counts": counts.to_json(),
            }
        )
        design = load_design(args, target_labels)
        top20_cols = get_top20_cols(args.top_features_csv, design)
        results, predictions, deltas = evaluate_target(
            df=design,
            target_name=f"{target_col}_within_suite_q25_q75",
            top20_cols=top20_cols,
            rng=rng,
            args=args,
        )
        all_results.append(results)
        all_predictions.append(predictions)
        if not deltas.empty:
            all_deltas.append(deltas)
        if target_col == "total_burden_rate":
            all_fdr.append(run_fdr(design, f"{target_col}_within_suite_q25_q75", float(args.fdr_alpha)))

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out_dir / "safety_burden_target_audit.csv", index=False)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    deltas_df = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    fdr_df = pd.concat(all_fdr, ignore_index=True) if all_fdr else pd.DataFrame()

    results_df.to_csv(out_dir / "safety_burden_quartile_results.csv", index=False)
    predictions_df.to_csv(out_dir / "safety_burden_quartile_predictions.csv", index=False)
    deltas_df.to_csv(out_dir / "conditional_safety_burden_delta.csv", index=False)
    fdr_df.to_csv(out_dir / "safety_burden_fdr.csv", index=False)

    prefix_df = pd.DataFrame()
    prefix_pred_df = pd.DataFrame()
    if not args.skip_prefix:
        total_labels = within_suite_quartile_labels(safety_stats, "total_burden_rate")
        top_feature_ids = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(20).tolist()
        prefix_df, prefix_pred_df = run_prefix_checks(labels=total_labels, top_feature_ids=top_feature_ids, args=args, rng=rng)
        prefix_df.to_csv(out_dir / "early_prefix_safety_burden_prediction.csv", index=False)
        prefix_pred_df.to_csv(out_dir / "early_prefix_safety_burden_predictions.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "rollout_dir": str(args.rollout_dir),
        "n_safety_episodes": int(len(safety_stats)),
        "episode_success_counts": safety_stats["episode_success"].value_counts(dropna=False).to_dict()
        if "episode_success" in safety_stats.columns
        else {},
        "mean_total_burden_rate": float(safety_stats["total_burden_rate"].mean()),
        "target_audit": audit_df.to_dict(orient="records"),
        "best_safety_burden_rows": results_df.sort_values("auroc", ascending=False).head(12).to_dict(orient="records")
        if len(results_df)
        else [],
        "conditional_delta_rows": deltas_df.to_dict(orient="records") if len(deltas_df) else [],
        "fdr_significant_features": int(fdr_df["significant"].astype(bool).sum())
        if len(fdr_df) and "significant" in fdr_df.columns
        else 0,
        "top_fdr_rows": fdr_df.head(20).to_dict(orient="records") if len(fdr_df) else [],
        "best_prefix_rows": prefix_df.sort_values("auroc", ascending=False).head(12).to_dict(orient="records")
        if len(prefix_df)
        else [],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
