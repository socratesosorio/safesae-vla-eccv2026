"""Generate cached-data confound controls for the ECCV rebuttal.

This script is intentionally self-contained and episode-level. It tests whether
the progress-feature result survives train-only feature selection, within-suite
relabeling, task/instruction controls, rich motion controls, episode-level FDR,
and random/permuted SAE feature controls.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


MISSING = "__missing__"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels_full_csv",
        type=str,
        default="logs/safesae_progress_labels/progress_labels_full.csv",
    )
    p.add_argument(
        "--sae_features_csv",
        type=str,
        default="logs/eccv_rebuttal_checks/episode_sae_layer20_d16384_means_all.csv",
    )
    p.add_argument(
        "--raw_features_csv",
        type=str,
        default="logs/eccv_rebuttal_checks/episode_raw_layer20_means.csv",
    )
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
    p.add_argument("--rollout_metadata_dir", type=str, default="data/rollouts")
    p.add_argument("--output_dir", type=str, default="logs/eccv_confound_controls")
    p.add_argument("--seed", type=int, default=12653)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--matched_random_trials", type=int, default=500)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
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


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]


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
    vals: list[float] = []
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


def paired_delta_ci(
    y_true: np.ndarray,
    base_scores: np.ndarray,
    new_scores: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    vals: list[float] = []
    y_true = np.asarray(y_true, dtype=int)
    base_scores = np.asarray(base_scores, dtype=float)
    new_scores = np.asarray(new_scores, dtype=float)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], new_scores[idx]) - metric_fn(y_true[idx], base_scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(np.asarray(vals), [2.5, 97.5]))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    n = float(len(ranked))
    adj = ranked * n / (np.arange(len(ranked), dtype=float) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[finite] = restored
    return out


def make_global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    df["label_scheme"] = "global_quartile"
    return df.reset_index(drop=True)


def make_within_suite_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for suite, part in labels_full.groupby("suite", sort=True):
        lo, hi = part["progress_norm"].quantile([0.25, 0.75])
        sub = part[(part["progress_norm"] <= lo) | (part["progress_norm"] >= hi)].copy()
        sub["label"] = (sub["progress_norm"] >= hi).astype(int)
        sub["label_scheme"] = "within_suite_quartile"
        rows.append(sub[["episode_id", "suite", "progress_norm", "label", "label_scheme"]])
    if not rows:
        return pd.DataFrame(columns=["episode_id", "suite", "progress_norm", "label", "label_scheme"])
    return pd.concat(rows, ignore_index=True)


def read_metadata(metadata_dir: Path, episode_ids: set[str]) -> pd.DataFrame:
    rows = []
    if not metadata_dir.exists():
        return pd.DataFrame(
            {
                "episode_id": sorted(episode_ids),
                "task_idx": [MISSING] * len(episode_ids),
                "instruction": [MISSING] * len(episode_ids),
            }
        )
    candidates = {p.stem: p for p in metadata_dir.rglob("rollout_*.json")}
    for ep in sorted(episode_ids):
        meta = {}
        path = candidates.get(ep)
        if path is not None:
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        rows.append(
            {
                "episode_id": ep,
                "task_idx": str(meta.get("task_idx", MISSING)),
                "instruction": str(meta.get("instruction", MISSING)),
                "metadata_suite": str(meta.get("suite", MISSING)),
            }
        )
    return pd.DataFrame(rows)


def prepare_design(
    labels: pd.DataFrame,
    sae_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    sae_use = sae_df.drop(columns=[c for c in ("label", "suite", "progress_norm") if c in sae_df.columns])
    tel_use = telemetry_df.drop(columns=[c for c in ("suite",) if c in telemetry_df.columns])
    df = (
        labels.merge(sae_use, on="episode_id", how="inner")
        .merge(raw_df, on="episode_id", how="inner")
        .merge(tel_use, on="episode_id", how="inner")
        .merge(metadata_df, on="episode_id", how="left")
    )
    df["suite"] = df["suite"].fillna(df.get("metadata_suite", MISSING)).fillna(MISSING).astype(str)
    for col in ["task_idx", "instruction"]:
        if col not in df.columns:
            df[col] = MISSING
        df[col] = df[col].fillna(MISSING).astype(str)
    return df.reset_index(drop=True)


def build_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    cat_cols: list[str],
    with_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    if numeric_cols:
        scaler = StandardScaler(with_mean=with_mean)
        x_train = train_df[numeric_cols].to_numpy(np.float32, copy=False)
        x_test = test_df[numeric_cols].to_numpy(np.float32, copy=False)
        train_parts.append(scaler.fit_transform(x_train).astype(np.float32, copy=False))
        test_parts.append(scaler.transform(x_test).astype(np.float32, copy=False))

    for col in cat_cols:
        train_vals = train_df[col].fillna(MISSING).astype(str)
        test_vals = test_df[col].fillna(MISSING).astype(str)
        cats = sorted(train_vals.unique().tolist())
        cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        x_train = np.zeros((len(train_df), len(cats)), dtype=np.float32)
        x_test = np.zeros((len(test_df), len(cats)), dtype=np.float32)
        for row_idx, val in enumerate(train_vals):
            x_train[row_idx, cat_to_idx[val]] = 1.0
        for row_idx, val in enumerate(test_vals):
            idx = cat_to_idx.get(val)
            if idx is not None:
                x_test[row_idx, idx] = 1.0
        train_parts.append(x_train)
        test_parts.append(x_test)

    if not train_parts:
        raise ValueError("No numeric or categorical columns supplied")
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def rank_features_by_train_effect(train_df: pd.DataFrame, feat_cols: list[str], y_col: str = "label") -> list[str]:
    y = train_df[y_col].to_numpy(dtype=int)
    x = train_df[feat_cols].to_numpy(np.float32, copy=False)
    low = x[y == 0]
    high = x[y == 1]
    if len(low) == 0 or len(high) == 0:
        return feat_cols[:]
    mean_low = low.mean(axis=0)
    mean_high = high.mean(axis=0)
    var_low = low.var(axis=0)
    var_high = high.var(axis=0)
    pooled = np.sqrt(0.5 * (var_low + var_high)) + 1e-8
    score = np.abs((mean_high - mean_low) / pooled)
    order = np.argsort(-score)
    return [feat_cols[int(i)] for i in order]


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
        yield 0, train_idx, test_idx
        return
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        yield fold, train_idx, test_idx


def cv_predict(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    selector: Callable[[pd.DataFrame], list[str]] | None = None,
    seed: int,
    folds: int,
    y_col: str = "label",
    method: str = "lr",
) -> tuple[pd.DataFrame, list[dict]]:
    numeric_cols = list(numeric_cols or [])
    cat_cols = list(cat_cols or [])
    y_all = df[y_col].to_numpy(dtype=int)
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, train_idx, test_idx in fold_iterator(y_all, folds=folds, seed=seed):
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            selected_cols = selector(train) if selector is not None else []
            selected_cols = list(selected_cols)
            use_numeric = numeric_cols + selected_cols
            x_train, x_test = build_matrix(
                train,
                test,
                numeric_cols=use_numeric,
                cat_cols=cat_cols,
                with_mean=False,
            )
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                solver="liblinear",
                random_state=int(seed),
            )
            clf.fit(x_train, train[y_col].to_numpy(dtype=int))
            scores = clf.decision_function(x_test)
            for i, score in zip(test_idx, scores):
                pred_rows.append(
                    {
                        "episode_id": df.iloc[int(i)]["episode_id"],
                        "fold": int(fold),
                        "y": int(df.iloc[int(i)][y_col]),
                        "score": float(score),
                    }
                )
            fold_rows.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "selected_features": ",".join(selected_cols),
                    "n_selected": int(len(selected_cols)),
                    "fold_auroc": safe_auroc(test[y_col].to_numpy(dtype=int), scores),
                    "fold_pr_auc": safe_pr_auc(test[y_col].to_numpy(dtype=int), scores),
                    "method": method,
                }
            )
    pred = pd.DataFrame(pred_rows).sort_values("episode_id").reset_index(drop=True)
    return pred, fold_rows


def cv_predict_raw_pca20(
    df: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    base_numeric_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    seed: int,
    folds: int,
    y_col: str = "label",
    method: str = "raw_pca20",
) -> tuple[pd.DataFrame, list[dict]]:
    base_numeric_cols = list(base_numeric_cols or [])
    cat_cols = list(cat_cols or [])
    y_all = df[y_col].to_numpy(dtype=int)
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, train_idx, test_idx in fold_iterator(y_all, folds=folds, seed=seed):
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            base_train, base_test = build_matrix(
                train,
                test,
                numeric_cols=base_numeric_cols,
                cat_cols=cat_cols,
                with_mean=False,
            ) if (base_numeric_cols or cat_cols) else (
                np.zeros((len(train), 0), dtype=np.float32),
                np.zeros((len(test), 0), dtype=np.float32),
            )
            raw_scaler = StandardScaler()
            raw_train = raw_scaler.fit_transform(train[raw_feature_cols].to_numpy(np.float32, copy=False))
            raw_test = raw_scaler.transform(test[raw_feature_cols].to_numpy(np.float32, copy=False))
            n_comp = min(20, raw_train.shape[1], len(train) - 1)
            pca = PCA(n_components=n_comp, random_state=int(seed))
            z_train = pca.fit_transform(raw_train).astype(np.float32, copy=False)
            z_test = pca.transform(raw_test).astype(np.float32, copy=False)
            x_train = np.concatenate([base_train, z_train], axis=1)
            x_test = np.concatenate([base_test, z_test], axis=1)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
                solver="liblinear",
                random_state=int(seed),
            )
            clf.fit(x_train, train[y_col].to_numpy(dtype=int))
            scores = clf.decision_function(x_test)
            for i, score in zip(test_idx, scores):
                pred_rows.append(
                    {
                        "episode_id": df.iloc[int(i)]["episode_id"],
                        "fold": int(fold),
                        "y": int(df.iloc[int(i)][y_col]),
                        "score": float(score),
                    }
                )
            fold_rows.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "selected_features": "",
                    "n_selected": int(n_comp),
                    "fold_auroc": safe_auroc(test[y_col].to_numpy(dtype=int), scores),
                    "fold_pr_auc": safe_pr_auc(test[y_col].to_numpy(dtype=int), scores),
                    "method": method,
                }
            )
    pred = pd.DataFrame(pred_rows).sort_values("episode_id").reset_index(drop=True)
    return pred, fold_rows


def summarize_predictions(
    name: str,
    pred: pd.DataFrame,
    *,
    scheme: str,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict:
    y = pred["y"].to_numpy(dtype=int)
    scores = pred["score"].to_numpy(dtype=float)
    auroc_ci = metric_ci(y, scores, metric_fn=safe_auroc, rng=rng, n_boot=bootstrap)
    pr_ci = metric_ci(y, scores, metric_fn=safe_pr_auc, rng=rng, n_boot=bootstrap)
    return {
        "scheme": scheme,
        "method": name,
        "n": int(len(pred)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "auroc": safe_auroc(y, scores),
        "auroc_ci95_low": auroc_ci[0],
        "auroc_ci95_high": auroc_ci[1],
        "pr_auc": safe_pr_auc(y, scores),
        "pr_auc_ci95_low": pr_ci[0],
        "pr_auc_ci95_high": pr_ci[1],
    }


def summarize_delta(
    *,
    scheme: str,
    base_method: str,
    new_method: str,
    base_pred: pd.DataFrame,
    new_pred: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict:
    merged = base_pred[["episode_id", "y", "score"]].merge(
        new_pred[["episode_id", "score"]], on="episode_id", suffixes=("_base", "_new")
    )
    y = merged["y"].to_numpy(dtype=int)
    base = merged["score_base"].to_numpy(dtype=float)
    new = merged["score_new"].to_numpy(dtype=float)
    auroc_delta = safe_auroc(y, new) - safe_auroc(y, base)
    pr_delta = safe_pr_auc(y, new) - safe_pr_auc(y, base)
    auroc_ci = paired_delta_ci(y, base, new, metric_fn=safe_auroc, rng=rng, n_boot=bootstrap)
    pr_ci = paired_delta_ci(y, base, new, metric_fn=safe_pr_auc, rng=rng, n_boot=bootstrap)
    return {
        "scheme": scheme,
        "base_method": base_method,
        "new_method": new_method,
        "n": int(len(merged)),
        "base_auroc": safe_auroc(y, base),
        "new_auroc": safe_auroc(y, new),
        "delta_auroc": float(auroc_delta),
        "delta_auroc_ci95_low": auroc_ci[0],
        "delta_auroc_ci95_high": auroc_ci[1],
        "base_pr_auc": safe_pr_auc(y, base),
        "new_pr_auc": safe_pr_auc(y, new),
        "delta_pr_auc": float(pr_delta),
        "delta_pr_auc_ci95_low": pr_ci[0],
        "delta_pr_auc_ci95_high": pr_ci[1],
    }


def nested_top20_analysis(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    submitted_top_cols: list[str],
    seed: int,
    folds: int,
    bootstrap: int,
) -> tuple[pd.DataFrame, list[dict], dict]:
    selector = lambda train: rank_features_by_train_effect(train, active_cols)[:20]
    pred, fold_rows = cv_predict(
        df,
        selector=selector,
        seed=seed,
        folds=folds,
        method="nested_top20_sae",
    )
    submitted = set(submitted_top_cols)
    for row in fold_rows:
        selected = [x for x in row["selected_features"].split(",") if x]
        row["overlap_with_submitted_top20"] = int(len(set(selected) & submitted))
    rng = np.random.default_rng(seed + 11)
    summary = summarize_predictions(
        "nested_top20_sae",
        pred,
        scheme=str(df["label_scheme"].iloc[0]),
        rng=rng,
        bootstrap=bootstrap,
    )
    summary["mean_fold_overlap_with_submitted_top20"] = float(
        np.mean([r["overlap_with_submitted_top20"] for r in fold_rows])
    )
    summary["fold_selected_features"] = [r["selected_features"] for r in fold_rows]
    return pred, fold_rows, summary


def evaluate_methods(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    submitted_top_cols: list[str],
    raw_feature_cols: list[str],
    motion_cols: list[str],
    seed: int,
    folds: int,
    bootstrap: int,
    scheme: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict]]:
    rng = np.random.default_rng(seed + len(scheme))
    preds: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    fold_details: list[dict] = []
    specs = [
        ("suite_id_only", [], ["suite"], None),
        ("task_id_only", [], ["task_idx"], None),
        ("instruction_id_only", [], ["instruction"], None),
        ("suite_task_instruction_id", [], ["suite", "task_idx", "instruction"], None),
        ("motion_telemetry", motion_cols, [], None),
        ("motion_suite_task", motion_cols, ["suite", "task_idx"], None),
        ("fixed_top20_sae", submitted_top_cols, [], None),
        ("full_sae", active_cols, [], None),
        ("raw_lr", raw_feature_cols, [], None),
    ]
    for method, nums, cats, selector in specs:
        pred, fold_rows = cv_predict(
            df,
            numeric_cols=nums,
            cat_cols=cats,
            selector=selector,
            seed=seed,
            folds=folds,
            method=method,
        )
        preds[method] = pred
        rows.append(summarize_predictions(method, pred, scheme=scheme, rng=rng, bootstrap=bootstrap))
        for fold_row in fold_rows:
            fold_row["scheme"] = scheme
        fold_details.extend(fold_rows)

    nested_pred, nested_folds, _nested_summary = nested_top20_analysis(
        df,
        active_cols=active_cols,
        submitted_top_cols=submitted_top_cols,
        seed=seed,
        folds=folds,
        bootstrap=bootstrap,
    )
    preds["nested_top20_sae"] = nested_pred
    rows.append(summarize_predictions("nested_top20_sae", nested_pred, scheme=scheme, rng=rng, bootstrap=bootstrap))
    for fold_row in nested_folds:
        fold_row["scheme"] = scheme
    fold_details.extend(nested_folds)

    pca_pred, pca_folds = cv_predict_raw_pca20(
        df,
        raw_feature_cols=raw_feature_cols,
        seed=seed,
        folds=folds,
        method="raw_pca20",
    )
    preds["raw_pca20"] = pca_pred
    rows.append(summarize_predictions("raw_pca20", pca_pred, scheme=scheme, rng=rng, bootstrap=bootstrap))
    for fold_row in pca_folds:
        fold_row["scheme"] = scheme
    fold_details.extend(pca_folds)
    return pd.DataFrame(rows), preds, fold_details


def conditional_motion_analysis(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    submitted_top_cols: list[str],
    raw_feature_cols: list[str],
    motion_cols: list[str],
    seed: int,
    folds: int,
    bootstrap: int,
    scheme: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000 + len(scheme))
    preds: dict[str, pd.DataFrame] = {}
    specs = {
        "motion_suite_task": (motion_cols, ["suite", "task_idx"], None),
        "motion_suite_task_top20": (motion_cols + submitted_top_cols, ["suite", "task_idx"], None),
        "motion_suite_task_full_sae": (motion_cols + active_cols, ["suite", "task_idx"], None),
        "motion_suite_task_nested_top20": (
            motion_cols,
            ["suite", "task_idx"],
            lambda train: rank_features_by_train_effect(train, active_cols)[:20],
        ),
    }
    for method, (nums, cats, selector) in specs.items():
        pred, _ = cv_predict(
            df,
            numeric_cols=nums,
            cat_cols=cats,
            selector=selector,
            seed=seed,
            folds=folds,
            method=method,
        )
        preds[method] = pred

    pca_pred, _ = cv_predict_raw_pca20(
        df,
        raw_feature_cols=raw_feature_cols,
        base_numeric_cols=motion_cols,
        cat_cols=["suite", "task_idx"],
        seed=seed,
        folds=folds,
        method="motion_suite_task_raw_pca20",
    )
    preds["motion_suite_task_raw_pca20"] = pca_pred

    rows = []
    base_name = "motion_suite_task"
    for method in [
        "motion_suite_task_top20",
        "motion_suite_task_nested_top20",
        "motion_suite_task_full_sae",
        "motion_suite_task_raw_pca20",
    ]:
        rows.append(
            summarize_delta(
                scheme=scheme,
                base_method=base_name,
                new_method=method,
                base_pred=preds[base_name],
                new_pred=preds[method],
                rng=rng,
                bootstrap=bootstrap,
            )
        )
    base_summary = summarize_predictions(
        base_name,
        preds[base_name],
        scheme=scheme,
        rng=rng,
        bootstrap=bootstrap,
    )
    base_summary["base_method"] = ""
    base_summary["new_method"] = base_name
    rows.insert(0, base_summary)
    return pd.DataFrame(rows)


def episode_level_fdr(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    submitted_top_cols: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, dict]:
    y = df["label"].to_numpy(dtype=int)
    low_df = df[y == 0]
    high_df = df[y == 1]
    rows = []
    for col in active_cols:
        low = low_df[col].to_numpy(np.float64, copy=False)
        high = high_df[col].to_numpy(np.float64, copy=False)
        if np.allclose(low, low[0] if len(low) else 0.0) and np.allclose(high, high[0] if len(high) else 0.0):
            p = 1.0
        else:
            try:
                p = float(mannwhitneyu(low, high, alternative="two-sided").pvalue)
            except ValueError:
                p = 1.0
        mean_low = float(np.mean(low)) if len(low) else float("nan")
        mean_high = float(np.mean(high)) if len(high) else float("nan")
        pooled = math.sqrt(0.5 * (float(np.var(low)) + float(np.var(high)))) + 1e-8
        effect = (mean_high - mean_low) / pooled
        rows.append(
            {
                "feature_idx": int(col[1:]),
                "feature_col": col,
                "p_value": p,
                "effect_size": float(effect),
                "abs_effect_size": float(abs(effect)),
                "mean_low_progress": mean_low,
                "mean_high_progress": mean_high,
                "direction": "higher_in_high_progress" if effect >= 0 else "higher_in_low_progress",
                "in_submitted_top20": bool(col in set(submitted_top_cols)),
            }
        )
    out = pd.DataFrame(rows)
    out["adjusted_p"] = bh_adjust(out["p_value"].to_numpy(dtype=float))
    out["significant"] = out["adjusted_p"] <= float(alpha)
    out = out.sort_values(["adjusted_p", "abs_effect_size"], ascending=[True, False]).reset_index(drop=True)
    sig = out[out["significant"]]
    top20_episode = set(out.head(20)["feature_col"].tolist())
    submitted = set(submitted_top_cols)
    summary = {
        "n_features_tested": int(len(out)),
        "n_significant": int(len(sig)),
        "submitted_top20_significant": int(out[out["feature_col"].isin(submitted)]["significant"].sum()),
        "episode_top20_overlap_with_submitted_top20": int(len(top20_episode & submitted)),
        "alpha": float(alpha),
    }
    return out, summary


def permutation_controls(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    seed: int,
    folds: int,
    n_perm: int,
    fdr_alpha: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2000)
    rows = []
    for trial in range(int(n_perm)):
        perm_df = df.copy()
        perm_y = perm_df["label"].to_numpy(dtype=int).copy()
        for _suite, idx in perm_df.groupby("suite").groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            perm_y[idx_arr] = rng.permutation(perm_y[idx_arr])
        perm_df["perm_label"] = perm_y
        selected = rank_features_by_train_effect(perm_df, active_cols, y_col="perm_label")[:20]
        pred, _ = cv_predict(
            perm_df,
            numeric_cols=selected,
            seed=seed + trial,
            folds=folds,
            y_col="perm_label",
            method="permuted_top20",
        )
        low = perm_df[perm_df["perm_label"] == 0][active_cols].to_numpy(np.float32, copy=False)
        high = perm_df[perm_df["perm_label"] == 1][active_cols].to_numpy(np.float32, copy=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            p = ttest_ind(low, high, axis=0, equal_var=False, nan_policy="omit").pvalue
        p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
        adj = bh_adjust(p)
        rows.append(
            {
                "trial": int(trial),
                "n": int(len(perm_df)),
                "top20_auroc": safe_auroc(pred["y"].to_numpy(dtype=int), pred["score"].to_numpy(dtype=float)),
                "top20_pr_auc": safe_pr_auc(pred["y"].to_numpy(dtype=int), pred["score"].to_numpy(dtype=float)),
                "welch_fdr_significant_count": int(np.sum(adj <= float(fdr_alpha))),
                "max_abs_train_effect": float(
                    np.max(
                        np.abs(
                            perm_df[perm_df["perm_label"] == 1][active_cols].mean(axis=0).to_numpy()
                            - perm_df[perm_df["perm_label"] == 0][active_cols].mean(axis=0).to_numpy()
                        )
                    )
                ),
                "selected_features": ",".join(selected),
            }
        )
    return pd.DataFrame(rows)


def matched_feature_set(
    *,
    top_cols: list[str],
    active_cols: list[str],
    stats_df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str]:
    selected: list[str] = []
    top_set = set(top_cols)
    pool = stats_df[stats_df["feature_col"].isin([c for c in active_cols if c not in top_set])].copy()
    for top_col in top_cols:
        target = stats_df[stats_df["feature_col"] == top_col]
        if target.empty:
            continue
        t = target.iloc[0]
        available = pool[~pool["feature_col"].isin(selected)].copy()
        if available.empty:
            break
        available["distance"] = (
            (available["activation_frequency"] - float(t["activation_frequency"])).abs()
            + (available["variance_rank"] - float(t["variance_rank"])).abs() / max(len(stats_df), 1)
            + (available["mean_activation"] - float(t["mean_activation"])).abs()
        )
        near = available.nsmallest(min(50, len(available)), "distance")
        selected.append(str(rng.choice(near["feature_col"].to_numpy())))
    if len(selected) < len(top_cols):
        remaining = [c for c in active_cols if c not in set(selected) and c not in top_set]
        fill = rng.choice(remaining, size=len(top_cols) - len(selected), replace=False).tolist()
        selected.extend([str(x) for x in fill])
    return selected[: len(top_cols)]


def matched_random_controls(
    df: pd.DataFrame,
    *,
    active_cols: list[str],
    submitted_top_cols: list[str],
    seed: int,
    folds: int,
    trials: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3000)
    x = df[active_cols].to_numpy(np.float32, copy=False)
    stats_df = pd.DataFrame(
        {
            "feature_col": active_cols,
            "activation_frequency": (x > 0).mean(axis=0),
            "mean_activation": x.mean(axis=0),
            "variance": x.var(axis=0),
        }
    )
    stats_df["variance_rank"] = stats_df["variance"].rank(method="average").to_numpy(dtype=float)
    rows = []
    true_pred, _ = cv_predict(
        df,
        numeric_cols=submitted_top_cols,
        seed=seed,
        folds=folds,
        method="submitted_top20",
    )
    rows.append(
        {
            "trial": -1,
            "control_type": "submitted_top20",
            "auroc": safe_auroc(true_pred["y"].to_numpy(dtype=int), true_pred["score"].to_numpy(dtype=float)),
            "pr_auc": safe_pr_auc(true_pred["y"].to_numpy(dtype=int), true_pred["score"].to_numpy(dtype=float)),
            "selected_features": ",".join(submitted_top_cols),
        }
    )
    for trial in range(int(trials)):
        selected = matched_feature_set(
            top_cols=submitted_top_cols,
            active_cols=active_cols,
            stats_df=stats_df,
            rng=rng,
        )
        pred, _ = cv_predict(
            df,
            numeric_cols=selected,
            seed=seed + trial,
            folds=folds,
            method="matched_random20",
        )
        rows.append(
            {
                "trial": int(trial),
                "control_type": "matched_random20",
                "auroc": safe_auroc(pred["y"].to_numpy(dtype=int), pred["score"].to_numpy(dtype=float)),
                "pr_auc": safe_pr_auc(pred["y"].to_numpy(dtype=int), pred["score"].to_numpy(dtype=float)),
                "selected_features": ",".join(selected),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_full = pd.read_csv(args.labels_full_csv)
    sae_df = pd.read_csv(args.sae_features_csv)
    raw_df = pd.read_csv(args.raw_features_csv)
    telemetry_df = pd.read_csv(args.telemetry_csv)
    top_df = pd.read_csv(args.top_features_csv)

    all_episode_ids = set(labels_full["episode_id"].astype(str))
    metadata_df = read_metadata(Path(args.rollout_metadata_dir), all_episode_ids)

    top_features = top_df["feature_idx"].astype(int).head(20).tolist()
    submitted_top_cols = [f"f{i}" for i in top_features if f"f{i}" in sae_df.columns]
    all_feat_cols = feature_cols(sae_df)
    global_labels = make_global_quartile_labels(labels_full)
    within_labels = make_within_suite_quartile_labels(labels_full)
    global_df = prepare_design(global_labels, sae_df, raw_df, telemetry_df, metadata_df)
    within_df = prepare_design(within_labels, sae_df, raw_df, telemetry_df, metadata_df)

    active_cols = [c for c in all_feat_cols if (global_df[c] > 0).any()]
    raw_feature_cols = raw_cols(raw_df)
    motion_cols = [
        c
        for c in [
            "eef_final_displacement",
            "eef_path_length",
            "eef_mean_velocity_proxy",
            "eef_max_velocity_proxy",
            "action_mean_norm",
            "action_max_norm",
            "action_translation_mean_norm",
            "contact_force_mean",
            "contact_force_max",
        ]
        if c in global_df.columns
    ]

    nested_pred, nested_folds, nested_summary = nested_top20_analysis(
        global_df,
        active_cols=active_cols,
        submitted_top_cols=submitted_top_cols,
        seed=seed,
        folds=int(args.folds),
        bootstrap=int(args.bootstrap),
    )
    nested_rows = pd.DataFrame(nested_folds)
    nested_overall = pd.DataFrame([{**nested_summary, "fold": -1}])
    pd.concat([nested_rows, nested_overall], ignore_index=True, sort=False).to_csv(
        out_dir / "nested_top20_cv.csv", index=False
    )

    within_results, _within_preds, fold_details = evaluate_methods(
        within_df,
        active_cols=active_cols,
        submitted_top_cols=submitted_top_cols,
        raw_feature_cols=raw_feature_cols,
        motion_cols=motion_cols,
        seed=seed,
        folds=int(args.folds),
        bootstrap=int(args.bootstrap),
        scheme="within_suite_quartile",
    )
    within_results.to_csv(out_dir / "within_suite_quartiles.csv", index=False)
    pd.DataFrame(fold_details).to_csv(out_dir / "within_suite_fold_details.csv", index=False)

    conditional_rows = []
    for scheme, df in [("global_quartile", global_df), ("within_suite_quartile", within_df)]:
        conditional_rows.append(
            conditional_motion_analysis(
                df,
                active_cols=active_cols,
                submitted_top_cols=submitted_top_cols,
                raw_feature_cols=raw_feature_cols,
                motion_cols=motion_cols,
                seed=seed,
                folds=int(args.folds),
                bootstrap=int(args.bootstrap),
                scheme=scheme,
            )
        )
    pd.concat(conditional_rows, ignore_index=True).to_csv(out_dir / "conditional_motion_delta.csv", index=False)

    fdr_df, fdr_summary = episode_level_fdr(
        global_df,
        active_cols=active_cols,
        submitted_top_cols=submitted_top_cols,
        alpha=float(args.fdr_alpha),
    )
    fdr_df.to_csv(out_dir / "episode_level_fdr.csv", index=False)

    perm_df = permutation_controls(
        global_df,
        active_cols=active_cols,
        seed=seed,
        folds=int(args.folds),
        n_perm=int(args.permutations),
        fdr_alpha=float(args.fdr_alpha),
    )
    perm_df.to_csv(out_dir / "permutation_controls.csv", index=False)

    matched_df = matched_random_controls(
        global_df,
        active_cols=active_cols,
        submitted_top_cols=submitted_top_cols,
        seed=seed,
        folds=int(args.folds),
        trials=int(args.matched_random_trials),
    )
    matched_df.to_csv(out_dir / "matched_random_controls.csv", index=False)

    summary = {
        "seed": seed,
        "n_global_quartile": int(len(global_df)),
        "global_quartile_counts": {
            "positives": int(global_df["label"].sum()),
            "negatives": int((1 - global_df["label"]).sum()),
        },
        "n_within_suite_quartile": int(len(within_df)),
        "within_suite_counts": {
            f"{suite}_label{label}": int(count)
            for (suite, label), count in within_df.groupby(["suite", "label"]).size().items()
        },
        "n_active_sae_features": int(len(active_cols)),
        "submitted_top_features": top_features,
        "motion_cols": motion_cols,
        "metadata": {
            "unique_tasks_global": int(global_df["task_idx"].nunique()),
            "unique_instructions_global": int(global_df["instruction"].nunique()),
            "metadata_dir": str(args.rollout_metadata_dir),
        },
        "nested_top20": nested_summary,
        "episode_level_fdr": fdr_summary,
        "permutation_controls": {
            "n_trials": int(len(perm_df)),
            "mean_null_top20_auroc": float(perm_df["top20_auroc"].mean()) if len(perm_df) else float("nan"),
            "p95_null_top20_auroc": float(perm_df["top20_auroc"].quantile(0.95)) if len(perm_df) else float("nan"),
            "mean_null_welch_fdr_count": float(perm_df["welch_fdr_significant_count"].mean()) if len(perm_df) else float("nan"),
        },
        "matched_random_controls": {
            "n_trials": int((matched_df["control_type"] == "matched_random20").sum()),
            "submitted_top20_auroc": float(matched_df[matched_df["control_type"] == "submitted_top20"]["auroc"].iloc[0]),
            "mean_matched_random20_auroc": float(matched_df[matched_df["control_type"] == "matched_random20"]["auroc"].mean()),
            "p95_matched_random20_auroc": float(matched_df[matched_df["control_type"] == "matched_random20"]["auroc"].quantile(0.95)),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
