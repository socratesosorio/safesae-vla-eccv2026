"""Generate final cached rebuttal add-ons.

This script covers three score-maximizing checks that do not require Athena:

1. Same-task / same-instruction pairwise progress ranking.
2. Signed feature-setting controls with frequency/mean/variance-matched random
   SAE features.
3. Early-prefix operating points with episode-bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_PATH = ROOT / "scripts" / "40_generate_eccv_temporal_phase_checks.py"
MISSING = "__missing__"


def _load_temporal_module():
    spec = importlib.util.spec_from_file_location("eccv_temporal_phase_checks", TEMPORAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TEMPORAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


temporal = _load_temporal_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--sae_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_sae_layer20_d16384_means_all.csv")
    p.add_argument("--raw_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_raw_layer20_means.csv")
    p.add_argument("--telemetry_csv", type=str, default="logs/eccv_rebuttal_checks/episode_telemetry_controls_and_semantic_audit.csv")
    p.add_argument("--submitted_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--robust_features_csv", type=str, default="logs/eccv_confound_controls_20260508-230421/episode_level_fdr.csv")
    p.add_argument("--rollout_metadata_dir", type=str, default="data/rollouts")
    p.add_argument("--rollout_tensor_dir", type=str, default="data/rollouts")
    p.add_argument("--sae_checkpoint", type=str, default="data/sae_checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_pairwise_signed_prefix_checks")
    p.add_argument("--progress_gap", type=float, default=0.20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--matched_trials", type=int, default=5000)
    p.add_argument("--match_pool_size", type=int, default=200)
    p.add_argument("--max_episodes", type=int, default=0, help="Optional row cap for smoke tests; 0 uses all rows.")
    p.add_argument("--prefixes", type=str, default="0.10,0.25")
    p.add_argument("--max_timesteps_per_prefix", type=int, default=16)
    p.add_argument("--skip_pairwise", action="store_true")
    p.add_argument("--skip_signed", action="store_true")
    p.add_argument("--skip_prefix", action="store_true")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]


def motion_cols(df: pd.DataFrame) -> list[str]:
    return [
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
        if c in df.columns
    ]


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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_metric_ci(
    y: np.ndarray,
    score: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    vals = []
    y = np.asarray(y)
    score = np.asarray(score)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        val = metric_fn(y[idx], score[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


def global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    return df.reset_index(drop=True)


def read_metadata(metadata_dir: Path, episode_ids: set[str]) -> pd.DataFrame:
    candidates = {}
    if metadata_dir.exists():
        candidates = {p.stem: p for p in metadata_dir.rglob("rollout_*.json")}
    rows = []
    for ep in sorted(episode_ids):
        meta = {}
        path = candidates.get(str(ep))
        if path is not None:
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        instruction = meta.get("instruction") or meta.get("task_description") or meta.get("language_instruction") or MISSING
        task_idx = meta.get("task_idx", meta.get("task_id", MISSING))
        suite = meta.get("suite", MISSING)
        rows.append(
            {
                "episode_id": str(ep),
                "meta_suite": str(suite),
                "task_idx": str(task_idx),
                "instruction": str(instruction),
            }
        )
    return pd.DataFrame(rows)


def load_design(args: argparse.Namespace, *, binary: bool) -> pd.DataFrame:
    labels_full = pd.read_csv(args.labels_full_csv)
    labels_full["episode_id"] = labels_full["episode_id"].astype(str)
    labels = global_quartile_labels(labels_full) if binary else labels_full[["episode_id", "suite", "progress_norm"]].copy()
    labels["episode_id"] = labels["episode_id"].astype(str)
    sae = pd.read_csv(args.sae_features_csv)
    raw = pd.read_csv(args.raw_features_csv)
    tel = pd.read_csv(args.telemetry_csv).drop(columns=["suite"], errors="ignore")
    for frame in [sae, raw, tel]:
        frame["episode_id"] = frame["episode_id"].astype(str)
    meta = read_metadata(Path(args.rollout_metadata_dir), set(labels["episode_id"]))
    df = labels.merge(sae, on="episode_id").merge(raw, on="episode_id").merge(tel, on="episode_id").merge(meta, on="episode_id", how="left")
    df["task_key"] = df["suite"].astype(str) + ":" + df["task_idx"].fillna(MISSING).astype(str)
    df["instruction_key"] = df["instruction"].fillna(MISSING).astype(str)
    if int(args.max_episodes) > 0 and len(df) > int(args.max_episodes):
        if binary and "label" in df.columns:
            parts = []
            per_class = max(1, int(args.max_episodes) // max(1, df["label"].nunique()))
            for _, part in df.groupby("label", sort=False):
                parts.append(part.sample(n=min(per_class, len(part)), random_state=int(args.seed)))
            df = pd.concat(parts, ignore_index=True)
            if len(df) < int(args.max_episodes):
                rest = labels.merge(sae, on="episode_id").merge(raw, on="episode_id").merge(tel, on="episode_id").merge(meta, on="episode_id", how="left")
                rest["task_key"] = rest["suite"].astype(str) + ":" + rest["task_idx"].fillna(MISSING).astype(str)
                rest["instruction_key"] = rest["instruction"].fillna(MISSING).astype(str)
                rest = rest[~rest["episode_id"].isin(df["episode_id"])]
                df = pd.concat(
                    [df, rest.sample(n=min(int(args.max_episodes) - len(df), len(rest)), random_state=int(args.seed))],
                    ignore_index=True,
                )
        else:
            df = df.sample(n=int(args.max_episodes), random_state=int(args.seed)).reset_index(drop=True)
    return df


def top_feature_cols(path: str, df: pd.DataFrame, n: int = 20) -> list[str]:
    top = pd.read_csv(path)
    cols = [f"f{int(x)}" for x in top["feature_idx"].head(n).tolist()]
    return [c for c in cols if c in df.columns and (df[c] > 0).any()]


def robust_feature_cols(path: str, df: pd.DataFrame, n: int = 20) -> list[str]:
    robust = pd.read_csv(path)
    robust = robust[robust.get("significant", True).astype(bool)] if "significant" in robust.columns else robust
    cols = [f"f{int(x)}" for x in robust["feature_idx"].head(n).tolist()]
    return [c for c in cols if c in df.columns and (df[c] > 0).any()]


def rank_train_top20(train_df: pd.DataFrame, cols: list[str], target: str) -> list[str]:
    y = train_df[target].to_numpy(dtype=float)
    x = train_df[cols].to_numpy(np.float32, copy=False)
    y = (y - y.mean()) / (y.std() + 1e-8)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-8
    corr = ((x - x_mean) * y[:, None]).mean(axis=0) / x_std
    return [cols[int(i)] for i in np.argsort(-np.abs(corr))[:20]]


def crossfit_ridge_scores(
    df: pd.DataFrame,
    *,
    method_specs: dict[str, tuple[str, list[str]]],
    active_cols: list[str],
    raw_feature_cols: list[str],
    seed: int,
    folds: int,
) -> pd.DataFrame:
    y = df["progress_norm"].to_numpy(dtype=float)
    kf = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    pred = {name: np.full(len(df), np.nan, dtype=float) for name in method_specs}
    selected_by_fold: dict[str, list[str]] = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        y_train = train["progress_norm"].to_numpy(dtype=float)
        for name, (kind, cols) in method_specs.items():
            use_cols = cols
            x_train: np.ndarray
            x_test: np.ndarray
            if kind == "nested_top20":
                use_cols = rank_train_top20(train, active_cols, "progress_norm")
                selected_by_fold[f"{name}_fold{fold}"] = use_cols
                x_train = train[use_cols].to_numpy(np.float32, copy=False)
                x_test = test[use_cols].to_numpy(np.float32, copy=False)
            elif kind == "raw_pca20":
                n_comp = min(20, len(raw_feature_cols), len(train_idx) - 1)
                model = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=seed), Ridge(alpha=10.0, solver="lsqr"))
                model.fit(train[raw_feature_cols].to_numpy(np.float32, copy=False), y_train)
                pred[name][test_idx] = model.predict(test[raw_feature_cols].to_numpy(np.float32, copy=False))
                continue
            else:
                x_train = train[use_cols].to_numpy(np.float32, copy=False)
                x_test = test[use_cols].to_numpy(np.float32, copy=False)
            model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0, solver="lsqr"))
            model.fit(x_train, y_train)
            pred[name][test_idx] = model.predict(x_test)
    rows = []
    for name, score in pred.items():
        rows.append(pd.DataFrame({"episode_id": df["episode_id"], "method": name, "score": score, "progress_norm": y}))
    return pd.concat(rows, ignore_index=True), selected_by_fold


def pairwise_eval(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    group_col: str,
    progress_gap: float,
    rng: np.random.Generator,
    bootstrap: int,
) -> pd.DataFrame:
    score_wide = scores.pivot(index="episode_id", columns="method", values="score")
    pair_rows = []
    for group, part in df.groupby(group_col, sort=False):
        if group == MISSING or len(part) < 2:
            continue
        for _, a in part.iterrows():
            pass
        for ia, ib in combinations(part.index.tolist(), 2):
            a = df.loc[ia]
            b = df.loc[ib]
            gap = float(b["progress_norm"] - a["progress_norm"])
            if abs(gap) < float(progress_gap):
                continue
            high_ep = b["episode_id"] if gap > 0 else a["episode_id"]
            low_ep = a["episode_id"] if gap > 0 else b["episode_id"]
            for method in score_wide.columns:
                hs = score_wide.loc[high_ep, method]
                ls = score_wide.loc[low_ep, method]
                if not np.isfinite(hs) or not np.isfinite(ls) or hs == ls:
                    continue
                pair_rows.append(
                    {
                        "group_col": group_col,
                        "group": group,
                        "method": method,
                        "high_episode": high_ep,
                        "low_episode": low_ep,
                        "progress_gap": abs(gap),
                        "score_delta_high_minus_low": float(hs - ls),
                        "correct": float(hs > ls),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pd.DataFrame(), pairs
    summary_rows = []
    for method, part in pairs.groupby("method", sort=False):
        vals = part["correct"].to_numpy(float)
        lo, hi = bootstrap_ci(vals, rng, bootstrap)
        summary_rows.append(
            {
                "group_col": group_col,
                "method": method,
                "n_pairs": int(len(part)),
                "n_groups": int(part["group"].nunique()),
                "pairwise_accuracy": float(vals.mean()),
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_progress_gap": float(part["progress_gap"].mean()),
                "mean_score_delta_high_minus_low": float(part["score_delta_high_minus_low"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), pairs


def run_pairwise(args: argparse.Namespace, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    df = load_design(args, binary=False)
    fcols = [c for c in feature_cols(df) if (df[c] > 0).any()]
    rcols = raw_cols(df)
    submitted = top_feature_cols(args.submitted_features_csv, df)
    robust = robust_feature_cols(args.robust_features_csv, df)
    mcols = motion_cols(df)
    specs = {
        "motion_ridge": ("standard", mcols),
        "submitted20_sae_ridge": ("standard", submitted),
        "robust20_sae_ridge": ("standard", robust),
        "nested_train_top20_sae_ridge": ("nested_top20", []),
        "full_sae_ridge": ("standard", fcols),
        "raw_pca20_ridge": ("raw_pca20", []),
    }
    scores, selected = crossfit_ridge_scores(
        df,
        method_specs=specs,
        active_cols=fcols,
        raw_feature_cols=rcols,
        seed=int(args.seed),
        folds=int(args.folds),
    )
    all_summaries = []
    all_pairs = []
    for group_col in ["task_key", "instruction_key"]:
        summary, pairs = pairwise_eval(
            df,
            scores,
            group_col=group_col,
            progress_gap=float(args.progress_gap),
            rng=rng,
            bootstrap=int(args.bootstrap),
        )
        all_summaries.append(summary)
        all_pairs.append(pairs)
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    scores.to_csv(out_dir / "same_task_pairwise_scores.csv", index=False)
    summary_df.to_csv(out_dir / "same_task_pairwise_ranking.csv", index=False)
    pairs_df.to_csv(out_dir / "same_task_pairwise_pairs.csv", index=False)
    (out_dir / "same_task_pairwise_selected_features.json").write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    return summary_df


def feature_stats(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    low = train[train["label"] == 0]
    high = train[train["label"] == 1]
    rows = []
    for col in cols:
        vals = train[col].to_numpy(float)
        rows.append(
            {
                "feature_col": col,
                "activation_rate": float((vals > 1e-8).mean()),
                "mean": float(vals.mean()),
                "variance": float(vals.var()),
                "gap_abs": float(abs(high[col].mean() - low[col].mean())),
            }
        )
    return pd.DataFrame(rows).set_index("feature_col")


def matched_feature_sample(
    top_cols: list[str],
    candidate_cols: list[str],
    stats: pd.DataFrame,
    rng: np.random.Generator,
    *,
    include_gap: bool,
    pool_size: int,
) -> list[str]:
    stat_cols = ["activation_rate", "mean", "variance"] + (["gap_abs"] if include_gap else [])
    mat = stats.loc[candidate_cols + top_cols, stat_cols].to_numpy(float)
    mu = np.nanmean(mat, axis=0)
    sig = np.nanstd(mat, axis=0) + 1e-12
    z = pd.DataFrame((stats[stat_cols] - mu) / sig, index=stats.index)
    chosen: list[str] = []
    remaining = set(candidate_cols)
    for top in top_cols:
        if not remaining:
            break
        d = ((z.loc[list(remaining)].to_numpy(float) - z.loc[top].to_numpy(float)[None, :]) ** 2).sum(axis=1)
        rem = np.asarray(list(remaining), dtype=object)
        order = np.argsort(d)
        pool = rem[order[: min(int(pool_size), len(order))]]
        pick = str(rng.choice(pool))
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def train_progress_probe(train: pd.DataFrame, active_cols: list[str], seed: int):
    scaler = StandardScaler(with_mean=False)
    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=int(seed))
    x_train = scaler.fit_transform(train[active_cols])
    y_train = train["label"].astype(int).to_numpy()
    probe.fit(x_train, y_train)
    scale = np.asarray(getattr(scaler, "scale_", np.ones(len(active_cols))), dtype=float)
    scale[scale == 0] = 1.0
    feature_weight = pd.Series(np.asarray(probe.coef_[0], dtype=float) / scale, index=active_cols)
    return scaler, probe, feature_weight


def patch_delta(base: pd.DataFrame, cols: list[str], values: pd.Series, feature_weight: pd.Series) -> np.ndarray:
    if not cols or len(base) == 0:
        return np.zeros(len(base), dtype=float)
    changed = values.loc[cols].to_numpy(float)[None, :] - base[cols].to_numpy(float)
    weights = feature_weight.loc[cols].to_numpy(float)
    return changed @ weights


def signed_control_rows(
    feature_set: str,
    cols: list[str],
    low_test: pd.DataFrame,
    high_test: pd.DataFrame,
    high_mean: pd.Series,
    low_mean: pd.Series,
    feature_weight: pd.Series,
) -> dict[str, np.ndarray]:
    return {
        f"{feature_set}_low_to_high": patch_delta(low_test, cols, high_mean, feature_weight),
        f"{feature_set}_high_to_low": patch_delta(high_test, cols, low_mean, feature_weight),
        f"{feature_set}_low_to_low_noop": patch_delta(low_test, cols, low_mean, feature_weight),
        f"{feature_set}_high_to_high_noop": patch_delta(high_test, cols, high_mean, feature_weight),
    }


def summarize_delta(name: str, values: np.ndarray, rng: np.random.Generator, bootstrap: int, empirical_p: float = float("nan")) -> dict[str, Any]:
    lo, hi = bootstrap_ci(values, rng, bootstrap)
    return {
        "condition": name,
        "n": int(len(values)),
        "mean_delta": float(np.mean(values)) if len(values) else float("nan"),
        "median_delta": float(np.median(values)) if len(values) else float("nan"),
        "ci95_low": lo,
        "ci95_high": hi,
        "frac_positive": float((values > 0).mean()) if len(values) else float("nan"),
        "frac_negative": float((values < 0).mean()) if len(values) else float("nan"),
        "empirical_p": float(empirical_p),
    }


def run_signed_controls(args: argparse.Namespace, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    df = load_design(args, binary=True)
    fcols = [c for c in feature_cols(df) if (df[c] > 0).any()]
    submitted = top_feature_cols(args.submitted_features_csv, df)
    robust = robust_feature_cols(args.robust_features_csv, df)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        random_state=int(args.seed),
        stratify=df["label"].astype(int).to_numpy(),
    )
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    low_test = test[test["label"] == 0].reset_index(drop=True)
    high_test = test[test["label"] == 1].reset_index(drop=True)
    scaler, probe, feature_weight = train_progress_probe(train, fcols, int(args.seed))
    test_score = probe.decision_function(scaler.transform(test[fcols]))
    high_mean = train[train["label"] == 1][fcols].mean(axis=0)
    low_mean = train[train["label"] == 0][fcols].mean(axis=0)
    stats = feature_stats(train, fcols)
    feature_sets = {"submitted20": submitted, "robust20_fdr": robust}
    rows = []
    random_rows = []
    for set_name, top_cols in feature_sets.items():
        top_cols = [c for c in top_cols if c in fcols][:20]
        if not top_cols:
            continue
        top_deltas = signed_control_rows(set_name, top_cols, low_test, high_test, high_mean, low_mean, feature_weight)
        candidate_cols = [c for c in fcols if c not in set(top_cols)]
        for match_name, include_gap in [("freq_mean_var", False), ("freq_mean_var_gap", True)]:
            random_means: dict[str, list[float]] = {k: [] for k in top_deltas}
            for trial in range(int(args.matched_trials)):
                cols = matched_feature_sample(
                    top_cols,
                    candidate_cols,
                    stats,
                    rng,
                    include_gap=include_gap,
                    pool_size=int(args.match_pool_size),
                )
                rand_deltas = signed_control_rows(f"{set_name}_{match_name}_random", cols, low_test, high_test, high_mean, low_mean, feature_weight)
                row = {"feature_set": set_name, "match": match_name, "trial": int(trial), "features": ",".join(cols)}
                for top_key, rand_key in zip(top_deltas.keys(), rand_deltas.keys()):
                    val = float(rand_deltas[rand_key].mean())
                    random_means[top_key].append(val)
                    row[top_key.replace(set_name, "random") + "_mean_delta"] = val
                random_rows.append(row)
            for key, vals in top_deltas.items():
                rand_vals = np.asarray(random_means[key], dtype=float)
                if key.endswith("low_to_high") or key.endswith("high_to_high_noop"):
                    p = (np.sum(rand_vals >= vals.mean()) + 1) / (len(rand_vals) + 1)
                else:
                    p = (np.sum(rand_vals <= vals.mean()) + 1) / (len(rand_vals) + 1)
                rec = summarize_delta(key, vals, rng, int(args.bootstrap), empirical_p=p)
                rec.update({"feature_set": set_name, "match": match_name, "k": int(len(top_cols)), "random_mean": float(rand_vals.mean())})
                rows.append(rec)
                rand_rec = summarize_delta(key.replace(set_name, f"{set_name}_{match_name}_random"), rand_vals, rng, int(args.bootstrap))
                rand_rec.update({"feature_set": set_name, "match": match_name, "k": int(len(top_cols)), "random_mean": float(rand_vals.mean())})
                rows.append(rand_rec)
    result = pd.DataFrame(rows)
    random_df = pd.DataFrame(random_rows)
    result.to_csv(out_dir / "matched_signed_feature_controls.csv", index=False)
    random_df.to_csv(out_dir / "matched_signed_feature_random_trials.csv", index=False)
    summary = {
        "probe_test_auroc": safe_auroc(test["label"].astype(int).to_numpy(), test_score),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_low_test": int(len(low_test)),
        "n_high_test": int(len(high_test)),
        "matched_trials": int(args.matched_trials),
    }
    (out_dir / "matched_signed_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return result


def op_from_predictions(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    pos = y == 1
    neg = ~pos
    tp = float((pred & pos).sum())
    fp = float((pred & neg).sum())
    return {
        "tpr": tp / max(float(pos.sum()), 1.0),
        "fpr": fp / max(float(neg.sum()), 1.0),
        "precision": tp / max(float(pred.sum()), 1.0),
    }


def prefix_op_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    train_y: np.ndarray,
    train_score: np.ndarray,
    fpr: float,
) -> tuple[float, np.ndarray]:
    neg_scores = train_score[train_y == 0]
    threshold = float(np.quantile(neg_scores, 1.0 - float(fpr)))
    return threshold, score >= threshold


def bootstrap_op_ci(y: np.ndarray, pred: np.ndarray, metric: str, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        vals.append(op_from_predictions(y[idx], pred[idx])[metric])
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


def fit_binary_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=int(seed)),
    )
    model.fit(train_x, train_y)
    return model.decision_function(train_x), model.decision_function(test_x)


def run_prefix_cis(args: argparse.Namespace, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    if args.skip_prefix:
        return pd.DataFrame()
    checkpoint = Path(args.sae_checkpoint)
    rollout_dir = Path(args.rollout_tensor_dir)
    if not checkpoint.exists() or not rollout_dir.exists():
        return pd.DataFrame()
    labels = global_quartile_labels(pd.read_csv(args.labels_full_csv))
    y_all = labels["label"].to_numpy(dtype=int)
    submitted = [int(x) for x in pd.read_csv(args.submitted_features_csv)["feature_idx"].head(20).tolist()]
    robust = [int(x) for x in pd.read_csv(args.robust_features_csv)["feature_idx"].head(20).tolist()]
    prefixes = temporal.parse_prefixes(args.prefixes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = temporal.load_sae_checkpoint(
        path=str(checkpoint),
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )
    rollout_map = {p.stem: p for p in rollout_dir.rglob("rollout_*.safetensors")}
    key = f"activations_layer{int(args.layer)}"
    packs = {prefix: {"episode_id": [], "y": [], "full": [], "motion": []} for prefix in prefixes}
    for _, row in labels.iterrows():
        path = rollout_map.get(str(row["episode_id"]))
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
        for prefix in prefixes:
            n = max(1, int(np.ceil(t * float(prefix))))
            idx = temporal.select_indices(n, int(args.max_timesteps_per_prefix))
            feats = temporal.encode_steps(sae, norm_factor, step_vecs[idx], device=device).mean(axis=0)
            packs[prefix]["episode_id"].append(str(row["episode_id"]))
            packs[prefix]["y"].append(int(row["label"]))
            packs[prefix]["full"].append(feats)
            packs[prefix]["motion"].append(temporal.motion_features(eef=eef, actions=actions, contact=contact, safety=safety, n=n))
    summary_rows = []
    pred_rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for prefix, pack in packs.items():
            yy = np.asarray(pack["y"], dtype=int)
            if len(yy) == 0:
                continue
            full = np.vstack(pack["full"]).astype(np.float32, copy=False)
            motion = np.vstack(pack["motion"]).astype(np.float32, copy=False)
            methods = {
                "submitted20_sae": full[:, submitted],
                "robust20_sae": full[:, robust],
                "motion": motion,
                "motion_plus_submitted20_sae": np.concatenate([motion, full[:, submitted]], axis=1),
                "motion_plus_robust20_sae": np.concatenate([motion, full[:, robust]], axis=1),
            }
            for method, x in methods.items():
                scores = np.full(len(yy), np.nan, dtype=float)
                pred_by_fpr = {0.10: np.zeros(len(yy), dtype=bool), 0.20: np.zeros(len(yy), dtype=bool)}
                for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed)).split(x, yy):
                    train_score, test_score = fit_binary_score(x[train_idx], yy[train_idx], x[test_idx], int(args.seed))
                    scores[test_idx] = test_score
                    for fpr_target in pred_by_fpr:
                        _, pred = prefix_op_metrics(yy[test_idx], test_score, yy[train_idx], train_score, fpr_target)
                        pred_by_fpr[fpr_target][test_idx] = pred
                au_ci = bootstrap_metric_ci(yy, scores, safe_auroc, rng, int(args.bootstrap))
                pr_ci = bootstrap_metric_ci(yy, scores, safe_pr_auc, rng, int(args.bootstrap))
                for fpr_target, preds in pred_by_fpr.items():
                    ops = op_from_predictions(yy, preds)
                    ci = {m: bootstrap_op_ci(yy, preds, m, rng, int(args.bootstrap)) for m in ["tpr", "fpr", "precision"]}
                    summary_rows.append(
                        {
                            "prefix_fraction": float(prefix),
                            "method": method,
                            "n": int(len(yy)),
                            "auroc": safe_auroc(yy, scores),
                            "auroc_ci95_low": au_ci[0],
                            "auroc_ci95_high": au_ci[1],
                            "pr_auc": safe_pr_auc(yy, scores),
                            "pr_auc_ci95_low": pr_ci[0],
                            "pr_auc_ci95_high": pr_ci[1],
                            "fpr_target": float(fpr_target),
                            "realized_fpr": ops["fpr"],
                            "realized_fpr_ci95_low": ci["fpr"][0],
                            "realized_fpr_ci95_high": ci["fpr"][1],
                            "tpr": ops["tpr"],
                            "tpr_ci95_low": ci["tpr"][0],
                            "tpr_ci95_high": ci["tpr"][1],
                            "precision": ops["precision"],
                            "precision_ci95_low": ci["precision"][0],
                            "precision_ci95_high": ci["precision"][1],
                        }
                    )
                    for ep, y, score, pred in zip(pack["episode_id"], yy, scores, preds):
                        pred_rows.append(
                            {
                                "prefix_fraction": float(prefix),
                                "method": method,
                                "fpr_target": float(fpr_target),
                                "episode_id": ep,
                                "label": int(y),
                                "score": float(score),
                                "predicted_positive": int(pred),
                            }
                        )
    summary = pd.DataFrame(summary_rows)
    pred = pd.DataFrame(pred_rows)
    summary.to_csv(out_dir / "early_prefix_operating_points_ci.csv", index=False)
    pred.to_csv(out_dir / "early_prefix_operating_point_predictions.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise = pd.DataFrame()
    signed = pd.DataFrame()
    prefix = pd.DataFrame()
    if not args.skip_pairwise:
        pairwise = run_pairwise(args, out_dir, rng)
    elif (out_dir / "same_task_pairwise_ranking.csv").exists():
        pairwise = pd.read_csv(out_dir / "same_task_pairwise_ranking.csv")
    if not args.skip_signed:
        signed = run_signed_controls(args, out_dir, rng)
    elif (out_dir / "matched_signed_feature_controls.csv").exists():
        signed = pd.read_csv(out_dir / "matched_signed_feature_controls.csv")
    if not args.skip_prefix:
        prefix = run_prefix_cis(args, out_dir, rng)
    elif (out_dir / "early_prefix_operating_points_ci.csv").exists():
        prefix = pd.read_csv(out_dir / "early_prefix_operating_points_ci.csv")
    summary = {
        "pairwise_best": pairwise.sort_values(["group_col", "pairwise_accuracy"], ascending=[True, False])
        .groupby("group_col")
        .head(6)
        .to_dict(orient="records")
        if not pairwise.empty
        else [],
        "signed_key_rows": signed[
            signed["condition"].str.contains("low_to_high|high_to_low", regex=True)
            & ~signed["condition"].str.contains("random", regex=False)
        ].to_dict(orient="records")
        if not signed.empty
        else [],
        "prefix_best": prefix.sort_values(["prefix_fraction", "fpr_target", "tpr"], ascending=[True, True, False])
        .groupby(["prefix_fraction", "fpr_target"])
        .head(5)
        .to_dict(orient="records")
        if not prefix.empty
        else [],
        "metadata": {
            "seed": int(args.seed),
            "progress_gap": float(args.progress_gap),
            "bootstrap": int(args.bootstrap),
            "matched_trials": int(args.matched_trials),
            "skip_pairwise": bool(args.skip_pairwise),
            "skip_signed": bool(args.skip_signed),
            "skip_prefix": bool(args.skip_prefix),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
