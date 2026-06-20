"""Run a compact 16K-vs-32K SAE dictionary sanity check for rebuttal.

This is intentionally small and rebuttal-focused. It compares the submitted
16K layer-20 SAE episode features against the existing 32K layer-20 checkpoint
on the same geometric-progress quartile labels.

The script encodes 32K episode feature means into the output directory only;
it does not overwrite the canonical logs or checkpoints.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="data/rollouts")
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument(
        "--sae16_features_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv",
    )
    p.add_argument("--sae32_checkpoint", type=str, default="data/sae_checkpoints/sae_layer20_d32768.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_32k_dictionary_sanity")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae32", type=int, default=32768)
    p.add_argument("--k32", type=int, default=48)
    p.add_argument("--max_timesteps_per_episode", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
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


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    df["episode_id"] = df["episode_id"].astype(str)
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    return df.reset_index(drop=True)


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
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), size=len(y))
        val = metric_fn(y[idx], scores[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


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


def fit_ridge_cv(x: np.ndarray, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(x[train_idx]).astype(np.float32, copy=False)
        x_test = scaler.transform(x[test_idx]).astype(np.float32, copy=False)
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=int(seed))
        clf.fit(x_train, y[train_idx])
        scores[test_idx] = clf.decision_function(x_test)
    return scores


def fit_lr_on_cols_cv(x: np.ndarray, y: np.ndarray, cols: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    x = np.asarray(x[:, cols], dtype=np.float32)
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
    return scores


def fit_nested_topk_cv(x: np.ndarray, y: np.ndarray, *, k: int, seed: int, folds: int) -> tuple[np.ndarray, list[list[int]]]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    scores = np.full((len(y),), np.nan, dtype=np.float64)
    selected: list[list[int]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in fold_iterator(y, folds=folds, seed=seed):
            x_train = x[train_idx]
            y_train = y[train_idx].astype(np.float32)
            centered_x = x_train - x_train.mean(axis=0, keepdims=True)
            centered_y = y_train - y_train.mean()
            denom = (x_train.std(axis=0) + 1e-8) * (y_train.std() + 1e-8)
            corr = (centered_x * centered_y[:, None]).mean(axis=0) / denom
            top_idx = np.argsort(-np.abs(corr))[: int(k)]
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
    dictionary: str,
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
        "dictionary": dictionary,
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


def sample_timestep_indices(num_steps: int, max_steps: int) -> np.ndarray:
    if num_steps <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_steps <= 0 or num_steps <= max_steps:
        return np.arange(num_steps, dtype=np.int64)
    return np.unique(np.linspace(0, num_steps - 1, num=max_steps, dtype=np.int64))


@torch.no_grad()
def encode_episode_means_32k(
    *,
    rollout_dir: Path,
    labels: pd.DataFrame,
    checkpoint: Path,
    layer: int,
    d_in: int,
    d_sae: int,
    k: int,
    max_timesteps_per_episode: int,
    output_npz: Path,
) -> tuple[list[str], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(str(checkpoint), d_in=d_in, d_sae=d_sae, k=k, device=device)
    key = f"activations_layer{int(layer)}"
    rollout_map = {p.stem: p for p in rollout_dir.rglob("rollout_*.safetensors")}
    episode_ids: list[str] = []
    features: list[np.ndarray] = []
    for ep in labels["episode_id"].astype(str).tolist():
        path = rollout_map.get(ep)
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        idx = sample_timestep_indices(step_vecs.shape[0], int(max_timesteps_per_episode))
        if len(idx) == 0:
            continue
        x = torch.from_numpy(step_vecs[idx].astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
        z = sae.encode(x).detach().cpu().numpy().astype(np.float32, copy=False)
        features.append(z.mean(axis=0))
        episode_ids.append(ep)
    if not features:
        raise RuntimeError("No 32K episode features were encoded.")
    matrix = np.vstack(features).astype(np.float32, copy=False)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, episode_id=np.asarray(episode_ids), features=matrix)
    return episode_ids, matrix


def load_16k_features(path: Path, labels: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df["episode_id"] = df["episode_id"].astype(str)
    df = df.drop(columns=[c for c in ("label", "suite", "progress_norm") if c in df.columns])
    merged = labels[["episode_id", "label"]].merge(df, on="episode_id", how="inner")
    cols = feature_cols(merged)
    return (
        merged["episode_id"].astype(str).tolist(),
        merged[cols].to_numpy(np.float32, copy=False),
        merged["label"].to_numpy(dtype=int),
    )


def align_32k(labels: pd.DataFrame, episode_ids: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    order = pd.DataFrame({"episode_id": episode_ids, "row": np.arange(len(episode_ids), dtype=int)})
    merged = labels[["episode_id", "label"]].merge(order, on="episode_id", how="inner")
    idx = merged["row"].to_numpy(dtype=int)
    return merged["episode_id"].astype(str).tolist(), matrix[idx], merged["label"].to_numpy(dtype=int)


def fdr_summary(x: np.ndarray, y: np.ndarray, dictionary: str, alpha: float) -> tuple[dict[str, Any], pd.DataFrame]:
    low = x[np.asarray(y, dtype=int) == 0]
    high = x[np.asarray(y, dtype=int) == 1]
    rows = []
    for j in range(x.shape[1]):
        low_vals = low[:, j]
        high_vals = high[:, j]
        if float(low_vals.max()) == 0.0 and float(high_vals.max()) == 0.0:
            continue
        try:
            stat, p_val = mannwhitneyu(low_vals, high_vals, alternative="two-sided")
        except ValueError:
            continue
        effect = 1.0 - (2.0 * float(stat)) / float(max(len(low_vals) * len(high_vals), 1))
        rows.append(
            {
                "dictionary": dictionary,
                "feature_idx": int(j),
                "u_statistic": float(stat),
                "p_value": float(p_val),
                "effect_size": float(effect),
                "abs_effect_size": float(abs(effect)),
                "mean_low_progress": float(low_vals.mean()),
                "mean_high_progress": float(high_vals.mean()),
                "freq_low_progress": float((low_vals > 0).mean()),
                "freq_high_progress": float((high_vals > 0).mean()),
                "direction": "higher_in_high_progress"
                if float(high_vals.mean()) >= float(low_vals.mean())
                else "higher_in_low_progress",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return {"dictionary": dictionary, "features_tested": 0, "significant_features": 0}, out
    reject, adj_p, _, _ = multipletests(out["p_value"], alpha=float(alpha), method="fdr_bh")
    out["adjusted_p"] = adj_p
    out["significant"] = reject
    out["composite_score"] = out["abs_effect_size"] * (-np.log10(out["adjusted_p"] + 1e-300))
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    summary = {
        "dictionary": dictionary,
        "features_tested": int(len(out)),
        "significant_features": int(out["significant"].astype(bool).sum()),
        "top10_mean_abs_effect": float(out["abs_effect_size"].head(10).mean()),
        "active_features": int((x.max(axis=0) > 0).sum()),
    }
    return summary, out


def pairwise_overlap(selected: list[list[int]]) -> float:
    if len(selected) < 2:
        return float("nan")
    vals = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            vals.append(len(set(selected[i]).intersection(selected[j])))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_dictionary(
    *,
    dictionary: str,
    x: np.ndarray,
    y: np.ndarray,
    top20_idx: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    full_scores = fit_ridge_cv(x, y, seed=int(args.seed), folds=int(args.folds))
    rows.append(
        summarize_scores(
            dictionary=dictionary,
            method="full_sae_ridge",
            y=y,
            scores=full_scores,
            rng=rng,
            bootstrap=int(args.bootstrap),
        )
    )
    nested_scores, selected = fit_nested_topk_cv(x, y, k=20, seed=int(args.seed), folds=int(args.folds))
    rows.append(
        summarize_scores(
            dictionary=dictionary,
            method="nested_train_top20_lr",
            y=y,
            scores=nested_scores,
            rng=rng,
            bootstrap=int(args.bootstrap),
            extra={"mean_pairwise_top20_overlap": pairwise_overlap(selected)},
        )
    )
    if len(top20_idx):
        fixed_scores = fit_lr_on_cols_cv(x, y, top20_idx, seed=int(args.seed), folds=int(args.folds))
        rows.append(
            summarize_scores(
                dictionary=dictionary,
                method="all_data_fdr_top20_lr_descriptive",
                y=y,
                scores=fixed_scores,
                rng=rng,
                bootstrap=int(args.bootstrap),
                extra={"leakage_note": "feature set ranked on all labeled episodes; descriptive only"},
            )
        )
    selected_payload = {"dictionary": dictionary, "fold_selected_features": selected}
    return rows, selected_payload


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    labels_full = pd.read_csv(args.labels_full_csv)
    labels = global_quartile_labels(labels_full)

    ids16, x16, y16 = load_16k_features(Path(args.sae16_features_csv), labels)
    npz_path = out_dir / "episode_feature_means_sae32768.npz"
    ids32, x32_all = encode_episode_means_32k(
        rollout_dir=Path(args.rollout_dir),
        labels=labels,
        checkpoint=Path(args.sae32_checkpoint),
        layer=int(args.layer),
        d_in=int(args.d_in),
        d_sae=int(args.d_sae32),
        k=int(args.k32),
        max_timesteps_per_episode=int(args.max_timesteps_per_episode),
        output_npz=npz_path,
    )
    ids32, x32, y32 = align_32k(labels, ids32, x32_all)

    fdr16_summary, fdr16 = fdr_summary(x16, y16, "16K_d16384_k32", float(args.fdr_alpha))
    fdr32_summary, fdr32 = fdr_summary(x32, y32, "32K_d32768_k48", float(args.fdr_alpha))
    top16 = fdr16["feature_idx"].head(20).to_numpy(dtype=int) if len(fdr16) else np.asarray([], dtype=int)
    top32 = fdr32["feature_idx"].head(20).to_numpy(dtype=int) if len(fdr32) else np.asarray([], dtype=int)

    rows16, selected16 = evaluate_dictionary(
        dictionary="16K_d16384_k32",
        x=x16,
        y=y16,
        top20_idx=top16,
        rng=rng,
        args=args,
    )
    rows32, selected32 = evaluate_dictionary(
        dictionary="32K_d32768_k48",
        x=x32,
        y=y32,
        top20_idx=top32,
        rng=rng,
        args=args,
    )
    results = pd.DataFrame(rows16 + rows32)
    fdr_summary_df = pd.DataFrame([fdr16_summary, fdr32_summary])
    top_features = pd.concat([fdr16.head(50), fdr32.head(50)], ignore_index=True) if len(fdr16) or len(fdr32) else pd.DataFrame()

    results.to_csv(out_dir / "dictionary_sanity_results.csv", index=False)
    fdr_summary_df.to_csv(out_dir / "dictionary_fdr_summary.csv", index=False)
    top_features.to_csv(out_dir / "dictionary_top_features.csv", index=False)
    pd.DataFrame({"episode_id": ids32}).to_csv(out_dir / "episode_feature_means_sae32768_index.csv", index=False)
    write_json(out_dir / "nested_selected_features.json", {"16K": selected16, "32K": selected32})

    summary = {
        "seed": int(args.seed),
        "n_16k": int(len(y16)),
        "n_32k": int(len(y32)),
        "labels": {
            "16k_positive": int(y16.sum()),
            "16k_negative": int((1 - y16).sum()),
            "32k_positive": int(y32.sum()),
            "32k_negative": int((1 - y32).sum()),
        },
        "max_timesteps_per_episode": int(args.max_timesteps_per_episode),
        "sae32_feature_npz": str(npz_path),
        "results": results.to_dict(orient="records"),
        "fdr_summary": fdr_summary_df.to_dict(orient="records"),
        "top16_features": top16.tolist(),
        "top32_features": top32.tolist(),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
