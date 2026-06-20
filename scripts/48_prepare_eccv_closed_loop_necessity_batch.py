"""Prepare a focused closed-loop feature-necessity batch.

Prior repair-style interventions were behaviorally active but not
top-feature-specific because random feature sets often produced similar shifts.
This batch asks the cleaner causal question: are the high-progress-associated
submitted features necessary for otherwise successful rollouts?

We set high-progress submitted features to low-progress class means or zero and
compare against activation/prevalence-matched random feature sets. Include only
if top-feature degradation is larger than matched random controls.
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
    p.add_argument("--out_dir", type=str, default="logs/eccv_closed_loop_necessity_batch")
    p.add_argument("--num_random_controls", type=int, default=6)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _class_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    feat_cols = _feature_cols(df)
    rows = []
    low_df = df[df["label"] == 0]
    high_df = df[df["label"] == 1]
    for col in feat_cols:
        idx = int(col[1:])
        low = float(low_df[col].mean())
        high = float(high_df[col].mean())
        rows.append(
            {
                "feature_idx": idx,
                "mean_low_progress": low,
                "mean_high_progress": high,
                "delta_high_minus_low": high - low,
                "active_rate": float((df[col] > 1e-8).mean()),
                "high_active_rate": float((high_df[col] > 1e-8).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("feature_idx")


def _load_high_progress_submitted(path: str, stats: pd.DataFrame) -> list[int]:
    top = pd.read_csv(path).head(20)
    out = []
    for feat in top["feature_idx"].astype(int).tolist():
        if int(feat) in stats.index and float(stats.loc[int(feat), "delta_high_minus_low"]) > 0:
            out.append(int(feat))
    if not out:
        raise RuntimeError("No high-progress-associated submitted features found")
    return out


def _matched_random_sets(stats: pd.DataFrame, target_features: list[int], *, n_sets: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(int(seed))
    target = stats.loc[target_features]
    excluded = set(int(x) for x in target_features)
    candidates = stats[
        (~stats.index.isin(excluded))
        & (stats["mean_high_progress"] > 0)
        & (stats["active_rate"] > 0)
    ].copy()
    if len(candidates) < len(target_features):
        raise RuntimeError(f"Only {len(candidates)} random candidates for {len(target_features)} target features")

    sets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(int(n_sets) * 20):
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
    vals = []
    for feat in features:
        if mode == "zero":
            val = 0.0
        elif mode == "lowmean":
            val = float(stats.loc[int(feat), "mean_low_progress"])
        else:
            raise ValueError(f"Unknown value mode {mode}")
        vals.append((int(feat), float(val)))
    return vals


def _write_manifest(
    path: Path,
    *,
    feature_set: str,
    values: list[tuple[int, float]],
    task_spec: str,
    condition_name: str,
    is_random: bool,
    value_mode: str,
) -> None:
    rows = []
    for rank, (feature_idx, feature_value) in enumerate(values):
        rows.append(
            {
                "feature_set": feature_set,
                "rank": int(rank),
                "feature_idx": int(feature_idx),
                "feature_value": float(feature_value),
                "default_scale": 1.0,
                "selection_strategy": "closed_loop_necessity_matched_random" if is_random else "closed_loop_necessity_submitted_high_progress",
                "intervention_direction": f"set_high_progress_features_to_{value_mode}",
                "is_random_control": int(bool(is_random)),
                "trigger_mode": "always",
                "trigger_start_step": 0,
                "trigger_latch": 1,
                "allowed_task_specs": task_spec,
                "condition_names": condition_name,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    stats = _class_stats(args.episode_features_csv)
    high_features = _load_high_progress_submitted(args.submitted_top_csv, stats)
    random_sets = _matched_random_sets(
        stats,
        high_features,
        n_sets=int(args.num_random_controls),
        seed=int(args.seed),
    )
    (out_dir / "selected_features.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_features.txt").write_text(
        "\n".join(str(x) for x in high_features) + "\n",
        encoding="utf-8",
    )

    jobs: list[dict[str, str | int]] = []

    def add_family(*, value_mode: str, task_spec: str, condition_name: str, n: int) -> None:
        task_name = task_spec.replace(":", "").replace(",", "_")
        base = f"submitted_high{len(high_features)}_{task_name}_{condition_name}_{value_mode}_n{n}"
        manifest = manifest_dir / f"{base}.csv"
        _write_manifest(
            manifest,
            feature_set=base,
            values=_values(high_features, stats, value_mode),
            task_spec=task_spec,
            condition_name=condition_name,
            is_random=False,
            value_mode=value_mode,
        )
        jobs.append(
            {
                "job_name": base,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_closed_loop_necessity_batch/{base}",
                "output_name": f"{base}.json",
            }
        )
        for i, feats in enumerate(random_sets):
            name = f"{base}_random{i:02d}"
            random_manifest = manifest_dir / f"{name}.csv"
            _write_manifest(
                random_manifest,
                feature_set=name,
                values=_values(feats, stats, value_mode),
                task_spec=task_spec,
                condition_name=condition_name,
                is_random=True,
                value_mode=value_mode,
            )
            jobs.append(
                {
                    "job_name": name,
                    "manifest": str(random_manifest),
                    "num_rollouts": int(n),
                    "output_dir": f"results/eccv_closed_loop_necessity_batch/{name}",
                    "output_name": f"{name}.json",
                }
            )

    # Goal:1 has stable baseline success in the prior closed-loop batches.
    add_family(value_mode="zero", task_spec="goal:1", condition_name="clean", n=96)
    add_family(value_mode="lowmean", task_spec="goal:1", condition_name="clean", n=96)
    # Add a small mixed semantic-target setting in case object-state tasks are more sensitive.
    add_family(value_mode="zero", task_spec="goal:1,object:0", condition_name="clean", n=120)

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'} using {len(high_features)} features")


if __name__ == "__main__":
    main()
