"""Prepare a focused closed-loop specificity rescue batch.

The necessity batch produced a surprising preliminary result: setting the
submitted high-progress-associated features to the low-progress class mean
improved goal:1 behavior. This batch asks whether that effect is feature/value
specific rather than a generic SAE perturbation:

1. Same submitted features, opposite class mean.
2. Same submitted features, short pulse instead of always-on.
3. Same submitted features, shuffled target values.
4. Activation/prevalence-matched random feature controls.

Include only if the submitted-feature condition separates cleanly from these
controls; otherwise keep the rebuttal's current cautious closed-loop language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submitted_top_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--episode_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--out_dir", type=str, default="logs/eccv_closed_loop_specificity_rescue")
    p.add_argument("--num_random_controls", type=int, default=4)
    p.add_argument("--num_shuffle_controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _class_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    feat_cols = _feature_cols(df)
    low_df = df[df["label"] == 0]
    high_df = df[df["label"] == 1]
    rows = []
    for col in feat_cols:
        idx = int(col[1:])
        low = float(low_df[col].mean())
        high = float(high_df[col].mean())
        rows.append(
            {
                "feature_idx": idx,
                "mean_low_progress": low,
                "mean_high_progress": high,
                "mid_progress": 0.5 * (low + high),
                "delta_high_minus_low": high - low,
                "active_rate": float((df[col] > 1e-8).mean()),
                "high_active_rate": float((high_df[col] > 1e-8).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("feature_idx")


def _load_submitted_high_delta(path: str, stats: pd.DataFrame) -> list[int]:
    top = pd.read_csv(path).head(20)
    features = []
    for feat in top["feature_idx"].astype(int).tolist():
        if int(feat) in stats.index and float(stats.loc[int(feat), "delta_high_minus_low"]) > 0:
            features.append(int(feat))
    if not features:
        raise RuntimeError("No submitted features with mean_high_progress > mean_low_progress")
    return features


def _matched_random_sets(stats: pd.DataFrame, target_features: list[int], *, n_sets: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(int(seed))
    target = stats.loc[target_features]
    candidates = stats[
        (~stats.index.isin(set(target_features)))
        & (stats["mean_high_progress"] > 0)
        & (stats["active_rate"] > 0)
    ].copy()
    sets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(int(n_sets) * 30):
        chosen: list[int] = []
        remaining = candidates.copy()
        shuffled_targets = target.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        for _, row in shuffled_targets.iterrows():
            log_high = np.log1p(float(row["mean_high_progress"]))
            act = float(row["active_rate"])
            score = (
                np.abs(np.log1p(remaining["mean_high_progress"].to_numpy()) - log_high)
                + 0.5 * np.abs(remaining["active_rate"].to_numpy() - act)
                + rng.normal(0, 0.005, size=len(remaining))
            )
            pick_pos = int(np.argmin(score))
            feat = int(remaining.index[pick_pos])
            chosen.append(feat)
            remaining = remaining.drop(index=feat)
        key = tuple(sorted(chosen))
        if key in seen:
            continue
        seen.add(key)
        sets.append(chosen)
        if len(sets) >= int(n_sets):
            break
    if len(sets) < int(n_sets):
        raise RuntimeError(f"Could only create {len(sets)} matched random sets")
    return sets


def _values(features: list[int], stats: pd.DataFrame, mode: str) -> list[tuple[int, float]]:
    column = {
        "lowmean": "mean_low_progress",
        "midmean": "mid_progress",
        "highmean": "mean_high_progress",
    }[mode]
    return [(int(feat), float(stats.loc[int(feat), column])) for feat in features]


def _write_manifest(
    path: Path,
    *,
    feature_set: str,
    values: list[tuple[int, float]],
    selection_strategy: str,
    intervention_direction: str,
    is_random: bool,
    trigger_start_step: int,
    trigger_end_step: int | None,
    trigger_latch: bool,
) -> None:
    rows = []
    for rank, (feature_idx, feature_value) in enumerate(values):
        row = {
            "feature_set": feature_set,
            "rank": int(rank),
            "feature_idx": int(feature_idx),
            "feature_value": float(feature_value),
            "default_scale": 1.0,
            "selection_strategy": selection_strategy,
            "intervention_direction": intervention_direction,
            "is_random_control": int(bool(is_random)),
            "trigger_mode": "after_step",
            "trigger_start_step": int(trigger_start_step),
            "trigger_latch": int(bool(trigger_latch)),
            "allowed_task_specs": "goal:1",
            "condition_names": "clean",
        }
        if trigger_end_step is not None:
            row["trigger_end_step"] = int(trigger_end_step)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    stats = _class_stats(args.episode_features_csv)
    submitted = _load_submitted_high_delta(args.submitted_top_csv, stats)
    random_sets = _matched_random_sets(
        stats,
        submitted,
        n_sets=int(args.num_random_controls),
        seed=int(args.seed),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_features.txt").write_text("\n".join(str(x) for x in submitted) + "\n", encoding="utf-8")

    jobs: list[dict[str, str | int]] = []

    def add_job(
        name: str,
        *,
        values: list[tuple[int, float]],
        selection_strategy: str,
        direction: str,
        is_random: bool = False,
        n: int = 64,
        start: int = 0,
        end: int | None = None,
        latch: bool = True,
    ) -> None:
        manifest = manifest_dir / f"{name}.csv"
        _write_manifest(
            manifest,
            feature_set=name,
            values=values,
            selection_strategy=selection_strategy,
            intervention_direction=direction,
            is_random=is_random,
            trigger_start_step=start,
            trigger_end_step=end,
            trigger_latch=latch,
        )
        jobs.append(
            {
                "job_name": name,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_closed_loop_specificity_rescue/{name}",
                "output_name": f"{name}.json",
            }
        )

    low_values = _values(submitted, stats, "lowmean")
    mid_values = _values(submitted, stats, "midmean")
    high_values = _values(submitted, stats, "highmean")

    add_job("goal1_top13_lowmean_start20_n64", values=low_values, selection_strategy="submitted_same_features", direction="set_to_low_progress_mean", start=20)
    add_job("goal1_top13_midmean_start20_n64", values=mid_values, selection_strategy="submitted_same_features", direction="set_to_mid_progress_mean", start=20)
    add_job("goal1_top13_highmean_start20_n64", values=high_values, selection_strategy="submitted_same_features", direction="set_to_high_progress_mean", start=20)
    add_job("goal1_top13_lowmean_pulse20_120_n64", values=low_values, selection_strategy="submitted_same_features", direction="set_to_low_progress_mean_short_pulse", start=20, end=120, latch=False)
    add_job("goal1_top13_highmean_pulse20_120_n64", values=high_values, selection_strategy="submitted_same_features", direction="set_to_high_progress_mean_short_pulse", start=20, end=120, latch=False)

    for i in range(int(args.num_shuffle_controls)):
        shuffled_values = [v for _, v in low_values]
        rng.shuffle(shuffled_values)
        add_job(
            f"goal1_top13_lowmean_shuffled_values_{i:02d}_n64",
            values=[(feat, float(val)) for feat, val in zip(submitted, shuffled_values)],
            selection_strategy="same_features_shuffled_values",
            direction="set_to_shuffled_low_progress_means",
            start=20,
            is_random=True,
        )

    for i, feats in enumerate(random_sets):
        add_job(
            f"goal1_random13_lowmean_start20_{i:02d}_n64",
            values=_values(feats, stats, "lowmean"),
            selection_strategy="matched_random_features",
            direction="set_to_low_progress_mean",
            start=20,
            is_random=True,
        )
        add_job(
            f"goal1_random13_lowmean_pulse20_120_{i:02d}_n64",
            values=_values(feats, stats, "lowmean"),
            selection_strategy="matched_random_features",
            direction="set_to_low_progress_mean_short_pulse",
            start=20,
            end=120,
            latch=False,
            is_random=True,
        )

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'} using {len(submitted)} submitted features")


if __name__ == "__main__":
    main()
