"""Generate task-heldout and continuous-progress rebuttal checks.

This script is cached-data only. It complements the confound-control package by
testing whether progress readouts survive held-out task/instruction groups and
whether the SAE signal tracks continuous relative geometric progress rather than
only an extreme quartile split.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


MISSING = "__missing__"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--sae_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_sae_layer20_d16384_means_all.csv")
    p.add_argument("--raw_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_raw_layer20_means.csv")
    p.add_argument("--telemetry_csv", type=str, default="logs/eccv_rebuttal_checks/episode_telemetry_controls_and_semantic_audit.csv")
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--rollout_metadata_dir", type=str, default="data/rollouts")
    p.add_argument("--output_dir", type=str, default="logs/eccv_generalization_continuous_checks")
    p.add_argument("--seed", type=int, default=12653)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
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


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)
    if finite.sum() < 3 or np.unique(y[finite]).size < 2 or np.unique(pred[finite]).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        return float(spearmanr(y[finite], pred[finite], nan_policy="omit").correlation)


def safe_pearson(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)
    if finite.sum() < 3 or np.unique(y[finite]).size < 2 or np.unique(pred[finite]).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        return float(pearsonr(y[finite], pred[finite])[0])


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    vals: list[float] = []
    y = np.asarray(y)
    pred = np.asarray(pred)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        val = metric_fn(y[idx], pred[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def make_global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    df["label_scheme"] = "global_quartile"
    return df.reset_index(drop=True)


def read_metadata(metadata_dir: Path, episode_ids: set[str]) -> pd.DataFrame:
    rows = []
    candidates: dict[str, Path] = {}
    if metadata_dir.exists():
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
    df["task_key"] = df["suite"].astype(str) + ":" + df["task_idx"].astype(str)
    df["instruction_key"] = df["instruction"].astype(str)
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
        for i, val in enumerate(train_vals):
            x_train[i, cat_to_idx[val]] = 1.0
        for i, val in enumerate(test_vals):
            j = cat_to_idx.get(val)
            if j is not None:
                x_test[i, j] = 1.0
        train_parts.append(x_train)
        test_parts.append(x_test)
    if not train_parts:
        raise ValueError("No features supplied")
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def rank_features_by_binary_effect(train_df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    y = train_df["label"].to_numpy(dtype=int)
    x = train_df[feat_cols].to_numpy(np.float32, copy=False)
    low, high = x[y == 0], x[y == 1]
    if len(low) == 0 or len(high) == 0:
        return feat_cols[:]
    pooled = np.sqrt(0.5 * (low.var(axis=0) + high.var(axis=0))) + 1e-8
    score = np.abs((high.mean(axis=0) - low.mean(axis=0)) / pooled)
    return [feat_cols[int(i)] for i in np.argsort(-score)]


def rank_features_by_continuous_corr(train_df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    y = train_df["progress_norm"].to_numpy(np.float32)
    x = train_df[feat_cols].to_numpy(np.float32, copy=False)
    y = y - y.mean()
    x = x - x.mean(axis=0, keepdims=True)
    denom = (np.sqrt((x * x).sum(axis=0)) * np.sqrt(float((y * y).sum()))) + 1e-8
    corr = np.abs((x * y[:, None]).sum(axis=0) / denom)
    return [feat_cols[int(i)] for i in np.argsort(-corr)]


def classification_splits(df: pd.DataFrame, *, group_col: str | None, folds: int, seed: int):
    y = df["label"].to_numpy(dtype=int)
    if group_col is None:
        n_splits = min(int(folds), int(np.bincount(y).min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        yield from splitter.split(np.zeros(len(df)), y)
        return

    groups = df[group_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return
    n_splits = min(int(folds), n_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for train_idx, test_idx in splitter.split(np.zeros(len(df)), y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        yield train_idx, test_idx


def regression_splits(df: pd.DataFrame, *, group_col: str | None, folds: int, seed: int):
    if group_col is None:
        splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
        yield from splitter.split(np.zeros(len(df)))
        return
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.array(sorted(np.unique(groups)))
    if len(unique_groups) < 2:
        return
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)
    n_splits = min(int(folds), len(unique_groups))
    chunks = np.array_split(unique_groups, n_splits)
    for chunk in chunks:
        test_mask = np.isin(groups, chunk)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


def fit_predict_classifier(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    method: str,
    numeric_cols: list[str],
    cat_cols: list[str],
    active_cols: list[str],
    raw_feature_cols: list[str],
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    selected: list[str] = []
    use_numeric = list(numeric_cols)
    if method == "nested_top20_sae":
        selected = rank_features_by_binary_effect(train, active_cols)[:20]
        use_numeric += selected
    if method == "raw_pca20":
        base_train, base_test = (
            build_matrix(train, test, numeric_cols=use_numeric, cat_cols=cat_cols, with_mean=False)
            if (use_numeric or cat_cols)
            else (np.zeros((len(train), 0), dtype=np.float32), np.zeros((len(test), 0), dtype=np.float32))
        )
        scaler = StandardScaler()
        raw_train = scaler.fit_transform(train[raw_feature_cols].to_numpy(np.float32, copy=False))
        raw_test = scaler.transform(test[raw_feature_cols].to_numpy(np.float32, copy=False))
        n_comp = min(20, raw_train.shape[1], len(train) - 1)
        pca = PCA(n_components=n_comp, random_state=int(seed))
        x_train = np.concatenate([base_train, pca.fit_transform(raw_train)], axis=1)
        x_test = np.concatenate([base_test, pca.transform(raw_test)], axis=1)
    else:
        x_train, x_test = build_matrix(train, test, numeric_cols=use_numeric, cat_cols=cat_cols, with_mean=False)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=int(seed))
    clf.fit(x_train, train["label"].to_numpy(dtype=int))
    return clf.decision_function(x_test), selected


def fit_predict_regressor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    method: str,
    numeric_cols: list[str],
    cat_cols: list[str],
    active_cols: list[str],
    raw_feature_cols: list[str],
    seed: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, list[str]]:
    selected: list[str] = []
    use_numeric = list(numeric_cols)
    if method == "nested_top20_sae":
        selected = rank_features_by_continuous_corr(train, active_cols)[:20]
        use_numeric += selected
    if method == "raw_pca20":
        base_train, base_test = (
            build_matrix(train, test, numeric_cols=use_numeric, cat_cols=cat_cols, with_mean=False)
            if (use_numeric or cat_cols)
            else (np.zeros((len(train), 0), dtype=np.float32), np.zeros((len(test), 0), dtype=np.float32))
        )
        scaler = StandardScaler()
        raw_train = scaler.fit_transform(train[raw_feature_cols].to_numpy(np.float32, copy=False))
        raw_test = scaler.transform(test[raw_feature_cols].to_numpy(np.float32, copy=False))
        n_comp = min(20, raw_train.shape[1], len(train) - 1)
        pca = PCA(n_components=n_comp, random_state=int(seed))
        x_train = np.concatenate([base_train, pca.fit_transform(raw_train)], axis=1)
        x_test = np.concatenate([base_test, pca.transform(raw_test)], axis=1)
    else:
        x_train, x_test = build_matrix(train, test, numeric_cols=use_numeric, cat_cols=cat_cols, with_mean=False)
    reg = Ridge(alpha=float(ridge_alpha), random_state=int(seed))
    reg.fit(x_train, train["progress_norm"].to_numpy(np.float32))
    return reg.predict(x_test), selected


def run_classification_cv(
    df: pd.DataFrame,
    *,
    split_name: str,
    group_col: str | None,
    method_specs: dict[str, tuple[list[str], list[str], str]],
    active_cols: list[str],
    raw_feature_cols: list[str],
    seed: int,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_idx, test_idx) in enumerate(classification_splits(df, group_col=group_col, folds=folds, seed=seed)):
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            for method, (num_cols, cat_cols, kind) in method_specs.items():
                scores, selected = fit_predict_classifier(
                    train,
                    test,
                    method=kind,
                    numeric_cols=num_cols,
                    cat_cols=cat_cols,
                    active_cols=active_cols,
                    raw_feature_cols=raw_feature_cols,
                    seed=seed,
                )
                y_test = test["label"].to_numpy(dtype=int)
                for ep, y, score in zip(test["episode_id"], y_test, scores):
                    pred_rows.append(
                        {
                            "split": split_name,
                            "method": method,
                            "fold": int(fold),
                            "episode_id": ep,
                            "y": int(y),
                            "score": float(score),
                        }
                    )
                fold_rows.append(
                    {
                        "split": split_name,
                        "method": method,
                        "fold": int(fold),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "train_groups": int(train[group_col].nunique()) if group_col else int(len(train)),
                        "test_groups": int(test[group_col].nunique()) if group_col else int(len(test)),
                        "test_positive_rate": float(y_test.mean()),
                        "fold_auroc": safe_auroc(y_test, scores),
                        "fold_pr_auc": safe_pr_auc(y_test, scores),
                        "selected_features": ",".join(selected),
                    }
                )
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def run_regression_cv(
    df: pd.DataFrame,
    *,
    split_name: str,
    group_col: str | None,
    method_specs: dict[str, tuple[list[str], list[str], str]],
    active_cols: list[str],
    raw_feature_cols: list[str],
    seed: int,
    folds: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(regression_splits(df, group_col=group_col, folds=folds, seed=seed)):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        for method, (num_cols, cat_cols, kind) in method_specs.items():
            pred, selected = fit_predict_regressor(
                train,
                test,
                method=kind,
                numeric_cols=num_cols,
                cat_cols=cat_cols,
                active_cols=active_cols,
                raw_feature_cols=raw_feature_cols,
                seed=seed,
                ridge_alpha=ridge_alpha,
            )
            y_test = test["progress_norm"].to_numpy(np.float32)
            for ep, y, score in zip(test["episode_id"], y_test, pred):
                pred_rows.append(
                    {
                        "split": split_name,
                        "method": method,
                        "fold": int(fold),
                        "episode_id": ep,
                        "progress_norm": float(y),
                        "prediction": float(score),
                    }
                )
            fold_rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "fold": int(fold),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_groups": int(train[group_col].nunique()) if group_col else int(len(train)),
                    "test_groups": int(test[group_col].nunique()) if group_col else int(len(test)),
                    "fold_spearman": safe_spearman(y_test, pred),
                    "fold_pearson": safe_pearson(y_test, pred),
                    "fold_r2": float(r2_score(y_test, pred)) if len(test) > 1 else float("nan"),
                    "fold_mae": float(mean_absolute_error(y_test, pred)),
                    "selected_features": ",".join(selected),
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def summarize_classification(pred: pd.DataFrame, *, rng: np.random.Generator, bootstrap: int) -> pd.DataFrame:
    rows = []
    for (split, method), group in pred.groupby(["split", "method"], sort=False):
        y = group["y"].to_numpy(dtype=int)
        score = group["score"].to_numpy(dtype=float)
        auroc_ci = bootstrap_ci(y, score, metric_fn=safe_auroc, rng=rng, n_boot=bootstrap)
        pr_ci = bootstrap_ci(y, score, metric_fn=safe_pr_auc, rng=rng, n_boot=bootstrap)
        rows.append(
            {
                "split": split,
                "method": method,
                "n": int(len(group)),
                "positives": int(y.sum()),
                "negatives": int((1 - y).sum()),
                "auroc": safe_auroc(y, score),
                "auroc_ci95_low": auroc_ci[0],
                "auroc_ci95_high": auroc_ci[1],
                "pr_auc": safe_pr_auc(y, score),
                "pr_auc_ci95_low": pr_ci[0],
                "pr_auc_ci95_high": pr_ci[1],
            }
        )
    return pd.DataFrame(rows)


def summarize_regression(pred: pd.DataFrame, *, rng: np.random.Generator, bootstrap: int) -> pd.DataFrame:
    rows = []
    for (split, method), group in pred.groupby(["split", "method"], sort=False):
        y = group["progress_norm"].to_numpy(dtype=float)
        score = group["prediction"].to_numpy(dtype=float)
        sp_ci = bootstrap_ci(y, score, metric_fn=safe_spearman, rng=rng, n_boot=bootstrap)
        pe_ci = bootstrap_ci(y, score, metric_fn=safe_pearson, rng=rng, n_boot=bootstrap)
        rows.append(
            {
                "split": split,
                "method": method,
                "n": int(len(group)),
                "spearman": safe_spearman(y, score),
                "spearman_ci95_low": sp_ci[0],
                "spearman_ci95_high": sp_ci[1],
                "pearson": safe_pearson(y, score),
                "pearson_ci95_low": pe_ci[0],
                "pearson_ci95_high": pe_ci[1],
                "r2": float(r2_score(y, score)) if len(group) > 1 else float("nan"),
                "mae": float(mean_absolute_error(y, score)),
            }
        )
    return pd.DataFrame(rows)


def residualize(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray, *, motion_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_test = build_matrix(train_df, test_df, numeric_cols=motion_cols, cat_cols=["suite", "task_idx"], with_mean=False)
    reg = Ridge(alpha=10.0)
    reg.fit(x_train, y_train)
    return y_train - reg.predict(x_train), y_test - reg.predict(x_test)


def residualized_continuous_correlations(
    df: pd.DataFrame,
    regression_pred: pd.DataFrame,
    *,
    motion_cols: list[str],
    seed: int,
    folds: int,
) -> pd.DataFrame:
    rows = []
    for split_name, group_col in [("episode_cv", None), ("task_heldout", "task_key"), ("instruction_heldout", "instruction_key")]:
        residual_rows = []
        for fold, (train_idx, test_idx) in enumerate(regression_splits(df, group_col=group_col, folds=folds, seed=seed)):
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            _, y_resid_test = residualize(
                train,
                test,
                train["progress_norm"].to_numpy(float),
                test["progress_norm"].to_numpy(float),
                motion_cols=motion_cols,
            )
            residual_rows.append(pd.DataFrame({"episode_id": test["episode_id"].to_numpy(), "target_residual": y_resid_test}))
        if not residual_rows:
            continue
        target_resid = pd.concat(residual_rows, ignore_index=True)
        split_pred = regression_pred[regression_pred["split"] == split_name]
        for method, part in split_pred.groupby("method", sort=False):
            merged = part[["episode_id", "prediction"]].merge(target_resid, on="episode_id", how="inner")
            if len(merged) < 3:
                continue
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "n": int(len(merged)),
                    "spearman_target_residual": safe_spearman(
                        merged["target_residual"].to_numpy(float),
                        merged["prediction"].to_numpy(float),
                    ),
                    "pearson_target_residual": safe_pearson(
                        merged["target_residual"].to_numpy(float),
                        merged["prediction"].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    labels_full = pd.read_csv(args.labels_full_csv)
    labels_full["episode_id"] = labels_full["episode_id"].astype(str)
    sae_df = pd.read_csv(args.sae_features_csv)
    raw_df = pd.read_csv(args.raw_features_csv)
    telemetry_df = pd.read_csv(args.telemetry_csv)
    for frame in [sae_df, raw_df, telemetry_df]:
        frame["episode_id"] = frame["episode_id"].astype(str)

    metadata_df = read_metadata(Path(args.rollout_metadata_dir), set(labels_full["episode_id"].astype(str)))
    top = pd.read_csv(args.top_features_csv)
    submitted_top_cols = [f"f{int(x)}" for x in top["feature_idx"].head(20).tolist()]

    continuous_labels = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    binary_labels = make_global_quartile_labels(labels_full)
    continuous_df = prepare_design(continuous_labels, sae_df, raw_df, telemetry_df, metadata_df)
    binary_df = prepare_design(binary_labels, sae_df, raw_df, telemetry_df, metadata_df)

    feat_cols = feature_cols(continuous_df)
    active_cols = [c for c in feat_cols if (continuous_df[c] > 0).any()]
    submitted_top_cols = [c for c in submitted_top_cols if c in continuous_df.columns]
    raw_feature_cols = raw_cols(continuous_df)
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
        if c in continuous_df.columns
    ]

    class_methods = {
        "suite_only": ([], ["suite"], "standard"),
        "motion": (motion_cols, [], "standard"),
        "motion_suite_task": (motion_cols, ["suite", "task_idx"], "standard"),
        "fixed_top20_sae": (submitted_top_cols, [], "standard"),
        "nested_top20_sae": ([], [], "nested_top20_sae"),
        "full_sae": (active_cols, [], "standard"),
        "raw_lr": (raw_feature_cols, [], "standard"),
        "raw_pca20": ([], [], "raw_pca20"),
    }
    reg_methods = {
        "suite_only": ([], ["suite"], "standard"),
        "motion": (motion_cols, [], "standard"),
        "motion_suite_task": (motion_cols, ["suite", "task_idx"], "standard"),
        "fixed_top20_sae": (submitted_top_cols, [], "standard"),
        "nested_top20_sae": ([], [], "nested_top20_sae"),
        "full_sae": (active_cols, [], "standard"),
        "raw_pca20": ([], [], "raw_pca20"),
    }

    class_preds = []
    class_folds = []
    for split_name, group_col in [
        ("episode_cv", None),
        ("task_heldout", "task_key"),
        ("instruction_heldout", "instruction_key"),
    ]:
        pred, folds = run_classification_cv(
            binary_df,
            split_name=split_name,
            group_col=group_col,
            method_specs=class_methods,
            active_cols=active_cols,
            raw_feature_cols=raw_feature_cols,
            seed=int(args.seed),
            folds=int(args.folds),
        )
        class_preds.append(pred)
        class_folds.append(folds)
    classification_predictions = pd.concat(class_preds, ignore_index=True)
    classification_fold_details = pd.concat(class_folds, ignore_index=True)
    classification_summary = summarize_classification(classification_predictions, rng=rng, bootstrap=int(args.bootstrap))

    reg_preds = []
    reg_folds = []
    for split_name, group_col in [
        ("episode_cv", None),
        ("task_heldout", "task_key"),
        ("instruction_heldout", "instruction_key"),
    ]:
        pred, folds = run_regression_cv(
            continuous_df,
            split_name=split_name,
            group_col=group_col,
            method_specs=reg_methods,
            active_cols=active_cols,
            raw_feature_cols=raw_feature_cols,
            seed=int(args.seed),
            folds=int(args.folds),
            ridge_alpha=float(args.ridge_alpha),
        )
        reg_preds.append(pred)
        reg_folds.append(folds)
    regression_predictions = pd.concat(reg_preds, ignore_index=True)
    regression_fold_details = pd.concat(reg_folds, ignore_index=True)
    regression_summary = summarize_regression(regression_predictions, rng=rng, bootstrap=int(args.bootstrap))
    residual_summary = residualized_continuous_correlations(
        continuous_df,
        regression_predictions,
        motion_cols=motion_cols,
        seed=int(args.seed),
        folds=int(args.folds),
    )

    classification_summary.to_csv(out_dir / "task_instruction_heldout_classification.csv", index=False)
    classification_fold_details.to_csv(out_dir / "task_instruction_heldout_classification_folds.csv", index=False)
    classification_predictions.to_csv(out_dir / "task_instruction_heldout_classification_predictions.csv", index=False)
    regression_summary.to_csv(out_dir / "continuous_progress_regression.csv", index=False)
    regression_fold_details.to_csv(out_dir / "continuous_progress_regression_folds.csv", index=False)
    regression_predictions.to_csv(out_dir / "continuous_progress_regression_predictions.csv", index=False)
    residual_summary.to_csv(out_dir / "continuous_progress_residualized_correlations.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "n_binary_episodes": int(len(binary_df)),
        "n_continuous_episodes": int(len(continuous_df)),
        "n_active_sae_features": int(len(active_cols)),
        "n_raw_features": int(len(raw_feature_cols)),
        "n_task_groups_binary": int(binary_df["task_key"].nunique()),
        "n_instruction_groups_binary": int(binary_df["instruction_key"].nunique()),
        "n_task_groups_continuous": int(continuous_df["task_key"].nunique()),
        "n_instruction_groups_continuous": int(continuous_df["instruction_key"].nunique()),
        "motion_cols": motion_cols,
        "submitted_top_cols": submitted_top_cols,
        "classification_best_rows": classification_summary.sort_values(["split", "auroc"], ascending=[True, False])
        .groupby("split")
        .head(3)
        .to_dict(orient="records"),
        "regression_best_rows": regression_summary.sort_values(["split", "spearman"], ascending=[True, False])
        .groupby("split")
        .head(3)
        .to_dict(orient="records"),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
