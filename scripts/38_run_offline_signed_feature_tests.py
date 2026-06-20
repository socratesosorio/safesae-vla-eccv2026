"""Run cached-data signed intervention tests for ECCV rebuttal triage.

These tests intentionally avoid simulator stochasticity. They ask whether the
ranked SAE features behave as signed progress directions under direct feature
setting/ablation in a held-out progress probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--output_dir", type=str, default="logs/eccv_offline_signed_feature_tests")
    p.add_argument("--top_ks", type=str, default="1,3,5,10,20")
    p.add_argument("--random_trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _feature_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("f") and c[1:].isdigit()]


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _patch(x: pd.DataFrame, cols: list[str], values: pd.Series) -> pd.DataFrame:
    y = x.copy()
    if cols:
        y.loc[:, cols] = values.loc[cols].to_numpy()[None, :]
    return y


def _delta(
    *,
    x_base: pd.DataFrame,
    x_patch: pd.DataFrame,
    cols: list[str],
    feature_weight: pd.Series,
) -> np.ndarray:
    if not cols:
        return np.zeros(len(x_base), dtype=np.float64)
    changed = x_patch[cols].to_numpy(np.float64) - x_base[cols].to_numpy(np.float64)
    weights = feature_weight.loc[cols].to_numpy(np.float64)
    return changed @ weights


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.episode_features_csv)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    top = pd.read_csv(args.top_features_csv)
    top_features = top["feature_idx"].astype(int).head(20).tolist()
    top_ks = [int(x) for x in str(args.top_ks).split(",") if x.strip()]

    feat_cols = _feature_cols(list(df.columns))
    active_cols = [c for c in feat_cols if (df[c] > 0).any()]
    top_cols_all = [f"f{i}" for i in top_features if f"f{i}" in df.columns]
    top_cols_all = [c for c in top_cols_all if c in active_cols]
    active_not_top = [c for c in active_cols if c not in set(top_cols_all)]

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

    scaler = StandardScaler(with_mean=False)
    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=int(args.seed))
    probe.fit(scaler.fit_transform(train[active_cols]), train["label"].astype(int).to_numpy())
    test_scores = probe.decision_function(scaler.transform(test[active_cols]))
    scale = np.asarray(getattr(scaler, "scale_", np.ones(len(active_cols))), dtype=np.float64)
    scale[scale == 0] = 1.0
    feature_weight = pd.Series(np.asarray(probe.coef_[0], dtype=np.float64) / scale, index=active_cols)

    high_mean = train[train["label"] == 1][feat_cols].mean(axis=0)
    low_mean = train[train["label"] == 0][feat_cols].mean(axis=0)
    zero_values = pd.Series(0.0, index=feat_cols)

    rows: list[dict[str, float | int | str]] = []
    random_rows: list[dict[str, float | int | str]] = []

    def add_condition(
        name: str,
        subset_name: str,
        base: pd.DataFrame,
        patched: pd.DataFrame,
        cols: list[str],
        k: int,
        random_p: float | None = None,
    ) -> np.ndarray:
        d = _delta(x_base=base, x_patch=patched, cols=cols, feature_weight=feature_weight)
        lo, hi = _bootstrap_ci(d, rng)
        rows.append(
            {
                "condition": name,
                "subset": subset_name,
                "k": int(k),
                "n": int(len(d)),
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "ci95_low": lo,
                "ci95_high": hi,
                "frac_positive": float((d > 0).mean()),
                "mean_abs_delta": float(np.abs(d).mean()),
                "empirical_p": float(random_p) if random_p is not None else float("nan"),
            }
        )
        return d

    for k in top_ks:
        top_cols = top_cols_all[:k]
        low_to_high = _patch(low_test, top_cols, high_mean)
        high_to_low = _patch(high_test, top_cols, low_mean)
        high_to_zero = _patch(high_test, top_cols, zero_values)

        low_delta = _delta(x_base=low_test, x_patch=low_to_high, cols=top_cols, feature_weight=feature_weight)
        high_low_delta = _delta(x_base=high_test, x_patch=high_to_low, cols=top_cols, feature_weight=feature_weight)
        high_zero_delta = _delta(x_base=high_test, x_patch=high_to_zero, cols=top_cols, feature_weight=feature_weight)

        rand_low_means = []
        rand_high_low_means = []
        rand_high_zero_means = []
        for trial in range(int(args.random_trials)):
            cols = list(rng.choice(active_not_top, size=k, replace=False))
            r_low = _delta(x_base=low_test, x_patch=_patch(low_test, cols, high_mean), cols=cols, feature_weight=feature_weight)
            r_high_low = _delta(x_base=high_test, x_patch=_patch(high_test, cols, low_mean), cols=cols, feature_weight=feature_weight)
            r_high_zero = _delta(x_base=high_test, x_patch=_patch(high_test, cols, zero_values), cols=cols, feature_weight=feature_weight)
            rand_low_means.append(float(r_low.mean()))
            rand_high_low_means.append(float(r_high_low.mean()))
            rand_high_zero_means.append(float(r_high_zero.mean()))
            random_rows.append(
                {
                    "trial": trial,
                    "k": int(k),
                    "low_to_high_mean_delta": float(r_low.mean()),
                    "high_to_low_mean_delta": float(r_high_low.mean()),
                    "high_to_zero_mean_delta": float(r_high_zero.mean()),
                }
            )

        rand_low = np.asarray(rand_low_means)
        rand_high_low = np.asarray(rand_high_low_means)
        rand_high_zero = np.asarray(rand_high_zero_means)
        p_low = (np.sum(rand_low >= low_delta.mean()) + 1) / (len(rand_low) + 1)
        p_high_low = (np.sum(rand_high_low <= high_low_delta.mean()) + 1) / (len(rand_high_low) + 1)
        p_high_zero = (np.sum(rand_high_zero <= high_zero_delta.mean()) + 1) / (len(rand_high_zero) + 1)

        add_condition("top_low_samples_to_high_mean", "low_progress", low_test, low_to_high, top_cols, k, p_low)
        add_condition("top_high_samples_to_low_mean", "high_progress", high_test, high_to_low, top_cols, k, p_high_low)
        add_condition("top_high_samples_to_zero", "high_progress", high_test, high_to_zero, top_cols, k, p_high_zero)

        for name, vals in [
            ("random_low_samples_to_high_mean", rand_low),
            ("random_high_samples_to_low_mean", rand_high_low),
            ("random_high_samples_to_zero", rand_high_zero),
        ]:
            lo, hi = _bootstrap_ci(vals, rng)
            rows.append(
                {
                    "condition": name,
                    "subset": "random_trial_means",
                    "k": int(k),
                    "n": int(len(vals)),
                    "mean_delta": float(vals.mean()),
                    "median_delta": float(np.median(vals)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "frac_positive": float((vals > 0).mean()),
                    "mean_abs_delta": float(np.abs(vals).mean()),
                    "empirical_p": float("nan"),
                }
            )

    result = pd.DataFrame(rows)
    random_df = pd.DataFrame(random_rows)
    result.to_csv(out_dir / "offline_signed_feature_tests.csv", index=False)
    random_df.to_csv(out_dir / "offline_signed_feature_random_trials.csv", index=False)

    summary = {
        "n_episodes": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_low_test": int(len(low_test)),
        "n_high_test": int(len(high_test)),
        "probe_test_auroc": _safe_auc(test["label"].astype(int).to_numpy(), test_scores),
        "random_trials": int(args.random_trials),
    }
    for k in top_ks:
        sub = result[(result["k"] == k) & result["condition"].str.startswith("top_")]
        for _, row in sub.iterrows():
            key = f"k{k}_{row['condition']}_mean_delta"
            summary[key] = float(row["mean_delta"])
            summary[f"k{k}_{row['condition']}_empirical_p"] = float(row["empirical_p"])
    (out_dir / "offline_signed_feature_tests_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
