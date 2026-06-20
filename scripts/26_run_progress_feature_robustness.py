"""Episode-level robustness checks for progress SAE features.

This script uses cached episode-level SAE feature means, so it avoids simulator
and activation-reencoding noise. It produces:

1. Patch-to-class-mean directionality on held-out low-progress episodes.
2. Top-feature activation prevalence and signed mean differences.
3. Leave-one-suite-out feature-ranking transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode_features_csv", type=str, required=True)
    p.add_argument("--labels_full_csv", type=str, required=True)
    p.add_argument("--top_features_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="logs/progress_feature_robustness")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--random_trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def feature_names(indices: list[int] | np.ndarray) -> list[str]:
    return [f"f{int(i)}" for i in indices]


def patch_to_values(x: pd.DataFrame, cols: list[str], values: pd.Series) -> pd.DataFrame:
    patched = x.copy()
    patched.loc[:, cols] = values.loc[cols].to_numpy()[None, :]
    return patched


def bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def patch_delta(
    *,
    x_base: pd.DataFrame,
    x_patch: pd.DataFrame,
    active_cols: list[str],
    scaler: StandardScaler,
    probe: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = probe.decision_function(scaler.transform(x_base[active_cols]))
    after = probe.decision_function(scaler.transform(x_patch[active_cols]))
    return base, after, after - base


def evaluate_patch(
    *,
    name: str,
    x_base: pd.DataFrame,
    x_patch: pd.DataFrame,
    active_cols: list[str],
    scaler: StandardScaler,
    probe: LogisticRegression,
) -> dict[str, float | str | int]:
    _, _, delta = patch_delta(x_base=x_base, x_patch=x_patch, active_cols=active_cols, scaler=scaler, probe=probe)
    return {
        "condition": name,
        "n_samples": int(len(x_base)),
        "mean_progress_logit_delta": float(np.mean(delta)),
        "median_progress_logit_delta": float(np.median(delta)),
        "frac_progress_logit_increased": float(np.mean(delta > 0)),
        "mean_abs_progress_logit_delta": float(np.mean(np.abs(delta))),
    }


def run_class_mean_patch(
    df: pd.DataFrame,
    top_features: list[int],
    *,
    top_k: int,
    random_trials: int,
    seed: int,
    output_dir: Path,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    feat_cols = feature_cols(df)
    active_cols = [c for c in feat_cols if (df[c] > 0).any()]
    top_cols = feature_names(top_features[:top_k])
    active_not_top = [c for c in active_cols if c not in set(top_cols)]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        random_state=seed,
        stratify=df["label"].to_numpy(),
    )
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    low_test = test[test["label"] == 0].reset_index(drop=True)

    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(train[active_cols])
    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed)
    probe.fit(x_train, train["label"].astype(int).to_numpy())
    test_auc = safe_auroc(test["label"].to_numpy(), probe.decision_function(scaler.transform(test[active_cols])))

    train_low_mean = train[train["label"] == 0][feat_cols].mean(axis=0)
    train_high_mean = train[train["label"] == 1][feat_cols].mean(axis=0)

    rows: list[dict[str, float | str | int]] = []
    top_low_patch = patch_to_values(low_test, top_cols, train_high_mean)
    top_base_logit, top_after_logit, top_sample_delta = patch_delta(
        x_base=low_test,
        x_patch=top_low_patch,
        active_cols=active_cols,
        scaler=scaler,
        probe=probe,
    )
    rows.append(
        evaluate_patch(
            name="top20_low_samples_to_high_class_mean",
            x_base=low_test,
            x_patch=top_low_patch,
            active_cols=active_cols,
            scaler=scaler,
            probe=probe,
        )
    )
    rows.append(
        evaluate_patch(
            name="top20_low_samples_to_low_class_mean",
            x_base=low_test,
            x_patch=patch_to_values(low_test, top_cols, train_low_mean),
            active_cols=active_cols,
            scaler=scaler,
            probe=probe,
        )
    )
    rows.append(
        evaluate_patch(
            name="top20_all_samples_to_high_class_mean",
            x_base=test,
            x_patch=patch_to_values(test, top_cols, train_high_mean),
            active_cols=active_cols,
            scaler=scaler,
            probe=probe,
        )
    )

    random_rows = []
    for trial in range(random_trials):
        cols = list(rng.choice(active_not_top, size=len(top_cols), replace=False))
        random_rows.append(
            evaluate_patch(
                name=f"random20_low_samples_to_high_class_mean_trial{trial:03d}",
                x_base=low_test,
                x_patch=patch_to_values(low_test, cols, train_high_mean),
                active_cols=active_cols,
                scaler=scaler,
                probe=probe,
            )
        )
    rand = pd.DataFrame(random_rows)
    rows.extend(random_rows)
    mean_row = {
        "condition": "random20_low_samples_to_high_class_mean_mean",
        "n_samples": int(len(low_test)),
    }
    for col in rand.columns:
        if col not in {"condition", "n_samples"}:
            mean_row[col] = float(rand[col].mean())
    rows.append(mean_row)

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "class_mean_patch_directionality.csv", index=False)

    sample_cols = ["episode_id", "suite", "label"]
    sample_rows = low_test[[c for c in sample_cols if c in low_test.columns]].copy()
    sample_rows["condition"] = "top20_low_samples_to_high_class_mean"
    sample_rows["base_progress_logit"] = top_base_logit
    sample_rows["patched_progress_logit"] = top_after_logit
    sample_rows["progress_logit_delta"] = top_sample_delta
    sample_rows.to_csv(output_dir / "class_mean_patch_sample_deltas.csv", index=False)

    top_delta = float(out.loc[out["condition"] == "top20_low_samples_to_high_class_mean", "mean_progress_logit_delta"].iloc[0])
    random_delta = rand["mean_progress_logit_delta"].to_numpy()
    top_ci = bootstrap_ci(top_sample_delta, rng=rng)
    random_ci = bootstrap_ci(random_delta, rng=rng)
    return {
        "patch_probe_test_auroc": float(test_auc),
        "patch_num_low_test": int(len(low_test)),
        "patch_top20_mean_logit_delta": top_delta,
        "patch_top20_mean_logit_delta_ci95_low": top_ci[0],
        "patch_top20_mean_logit_delta_ci95_high": top_ci[1],
        "patch_random_mean_logit_delta": float(np.mean(random_delta)),
        "patch_random_mean_logit_delta_ci95_low": random_ci[0],
        "patch_random_mean_logit_delta_ci95_high": random_ci[1],
        "patch_random_std_logit_delta": float(np.std(random_delta)),
        "patch_top20_minus_random_mean_logit_delta": float(top_delta - np.mean(random_delta)),
        "patch_empirical_p_random_ge_top20": float((np.sum(random_delta >= top_delta) + 1) / (len(random_delta) + 1)),
    }


def run_prevalence_table(df: pd.DataFrame, top_df: pd.DataFrame, *, top_k: int, output_dir: Path) -> None:
    rows = []
    for rank, row in enumerate(top_df.head(top_k).itertuples(index=False), start=1):
        idx = int(getattr(row, "feature_idx"))
        col = f"f{idx}"
        low = df[df["label"] == 0][col]
        high = df[df["label"] == 1][col]
        all_vals = df[col]
        rows.append(
            {
                "rank": rank,
                "feature_idx": idx,
                "active_rate_all": float((all_vals > 0).mean()),
                "active_rate_low": float((low > 0).mean()),
                "active_rate_high": float((high > 0).mean()),
                "mean_low_progress": float(low.mean()),
                "mean_high_progress": float(high.mean()),
                "mean_high_minus_low": float(high.mean() - low.mean()),
                "direction_by_mean": "higher_in_high_progress" if high.mean() > low.mean() else "higher_in_low_progress",
                "reported_direction": str(getattr(row, "direction", "")),
                "composite_score": float(getattr(row, "composite_score")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "top_feature_activation_prevalence.csv", index=False)


def rank_features(train: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    low = train[train["label"] == 0]
    high = train[train["label"] == 1]
    rows = []
    for col in feat_cols:
        x = low[col].to_numpy()
        y = high[col].to_numpy()
        mean_low = float(np.mean(x))
        mean_high = float(np.mean(y))
        pooled = float(np.sqrt(0.5 * (np.var(x) + np.var(y))) + 1e-8)
        effect = (mean_high - mean_low) / pooled
        try:
            p_val = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
        except Exception:
            p_val = 1.0
        rows.append(
            {
                "feature": col,
                "feature_idx": int(col[1:]),
                "mean_low": mean_low,
                "mean_high": mean_high,
                "signed_effect": effect,
                "abs_effect": abs(effect),
                "p_value": p_val,
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_effect", "p_value"], ascending=[False, True]).reset_index(drop=True)


def run_leave_one_suite_out(
    df: pd.DataFrame,
    top_features: list[int],
    *,
    top_k: int,
    random_trials: int,
    seed: int,
    output_dir: Path,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    feat_cols = feature_cols(df)
    active_cols = [c for c in feat_cols if (df[c] > 0).any()]
    global_top = set(top_features[:top_k])
    rows = []
    for suite in sorted(df["suite"].dropna().unique()):
        train = df[df["suite"] != suite].reset_index(drop=True)
        hold = df[df["suite"] == suite].reset_index(drop=True)
        if len(np.unique(train["label"])) < 2 or len(np.unique(hold["label"])) < 2:
            continue
        ranked = rank_features(train, active_cols)
        top = ranked.head(top_k).copy()
        top_cols = top["feature"].tolist()
        signs = np.sign(top["signed_effect"].to_numpy())
        signs[signs == 0] = 1.0
        score = hold[top_cols].to_numpy() @ signs / float(top_k)
        auc = safe_auroc(hold["label"].to_numpy(), score)

        hold_low = hold[hold["label"] == 0]
        hold_high = hold[hold["label"] == 1]
        hold_effects = []
        for col, sign in zip(top_cols, signs):
            diff = float(hold_high[col].mean() - hold_low[col].mean())
            hold_effects.append(np.sign(diff) == sign if abs(diff) > 1e-12 else False)
        active_pool = np.asarray(active_cols)
        random_aucs = []
        for _ in range(random_trials):
            cols = list(rng.choice(active_pool, size=top_k, replace=False))
            random_signs = rng.choice([-1.0, 1.0], size=top_k)
            random_score = hold[cols].to_numpy() @ random_signs / float(top_k)
            random_aucs.append(safe_auroc(hold["label"].to_numpy(), random_score))
        rows.append(
            {
                "heldout_suite": suite,
                "n_train": int(len(train)),
                "n_holdout": int(len(hold)),
                "n_holdout_low": int((hold["label"] == 0).sum()),
                "n_holdout_high": int((hold["label"] == 1).sum()),
                "signed_top20_auroc": float(auc),
                "random_signed20_auroc_mean": float(np.nanmean(random_aucs)),
                "random_signed20_auroc_std": float(np.nanstd(random_aucs)),
                "signed_top20_minus_random": float(auc - np.nanmean(random_aucs)),
                "top20_global_overlap": int(len(set(top["feature_idx"].astype(int).tolist()) & global_top)),
                "heldout_sign_agreement_frac": float(np.mean(hold_effects)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "leave_one_suite_out_ranking_stability.csv", index=False)
    return {
        "loso_mean_signed_top20_auroc": float(out["signed_top20_auroc"].mean()) if not out.empty else float("nan"),
        "loso_mean_top20_minus_random": float(out["signed_top20_minus_random"].mean()) if not out.empty else float("nan"),
        "loso_mean_sign_agreement": float(out["heldout_sign_agreement_frac"].mean()) if not out.empty else float("nan"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.episode_features_csv)
    labels = pd.read_csv(args.labels_full_csv)[["episode_id", "suite", "progress_norm"]]
    df = feats.merge(labels, on="episode_id", how="left")
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    top_df = pd.read_csv(args.top_features_csv)
    top_features = top_df["feature_idx"].astype(int).tolist()

    summary: dict[str, float | int] = {
        "num_labeled_episodes": int(len(df)),
        "num_low_progress": int((df["label"] == 0).sum()),
        "num_high_progress": int((df["label"] == 1).sum()),
    }
    summary.update(
        run_class_mean_patch(
            df,
            top_features,
            top_k=args.top_k,
            random_trials=args.random_trials,
            seed=args.seed,
            output_dir=out_dir,
        )
    )
    run_prevalence_table(df, top_df, top_k=args.top_k, output_dir=out_dir)
    summary.update(
        run_leave_one_suite_out(
            df,
            top_features,
            top_k=args.top_k,
            random_trials=args.random_trials,
            seed=args.seed,
            output_dir=out_dir,
        )
    )

    with (out_dir / "progress_feature_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
