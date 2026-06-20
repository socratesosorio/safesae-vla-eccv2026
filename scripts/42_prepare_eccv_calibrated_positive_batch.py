"""Prepare calibrated closed-loop jobs for ECCV rebuttal rescue.

The previous broad high-mean batch showed that always-on top-20 feature setting
can move policy behavior, but not in a feature-specific or consistently helpful
direction. This batch focuses on narrower, more defensible candidates:

1. Lower-gain alpha=1.0 settings that already looked less oversteered.
2. Early and short-pulse interventions to avoid late-trajectory side effects.
3. Mild-condition targeted goal:1 rollouts where there is room to help.
4. A robust FDR-ranked feature set from cached confound-control analysis.
5. Many matched random controls for every candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--submitted_top_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv",
    )
    p.add_argument(
        "--robust_fdr_csv",
        type=str,
        default="logs/eccv_confound_controls_20260508-230421/episode_level_fdr.csv",
    )
    p.add_argument(
        "--episode_features_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv",
    )
    p.add_argument("--out_dir", type=str, default="logs/eccv_calibrated_positive_batch")
    p.add_argument("--num_random_controls", type=int, default=8)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _class_stats_from_episode_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    feat_cols = _feature_cols(df)
    low = df[df["label"] == 0][feat_cols].mean(axis=0)
    high = df[df["label"] == 1][feat_cols].mean(axis=0)
    active = (df[feat_cols] > 0).any(axis=0)
    rows = []
    for col in feat_cols:
        rows.append(
            {
                "feature_idx": int(col[1:]),
                "feature_col": col,
                "mean_low_progress": float(low[col]),
                "mean_high_progress": float(high[col]),
                "active": bool(active[col]),
            }
        )
    return pd.DataFrame(rows).set_index("feature_idx")


def _load_submitted(path: str, stats: pd.DataFrame, k: int = 20) -> list[int]:
    df = pd.read_csv(path).head(k)
    return [int(x) for x in df["feature_idx"].tolist() if int(x) in stats.index]


def _load_robust(path: str, stats: pd.DataFrame, k: int = 20) -> list[int]:
    df = pd.read_csv(path)
    if "significant" in df.columns:
        df = df[df["significant"].astype(bool)]
    if "abs_effect_size" in df.columns:
        df = df.sort_values("abs_effect_size", ascending=False, kind="stable")
    elif "adjusted_p" in df.columns:
        df = df.sort_values("adjusted_p", ascending=True, kind="stable")
    return [int(x) for x in df["feature_idx"].tolist() if int(x) in stats.index][:k]


def _values(features: list[int], stats: pd.DataFrame, alpha: float) -> list[tuple[int, float]]:
    vals = []
    for feat in features:
        row = stats.loc[int(feat)]
        low = float(row["mean_low_progress"])
        high = float(row["mean_high_progress"])
        vals.append((int(feat), low + float(alpha) * (high - low)))
    return vals


def _random_sets(
    *,
    stats: pd.DataFrame,
    exclude: set[int],
    k: int,
    n_sets: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(int(seed))
    candidates = [
        int(idx)
        for idx, row in stats.iterrows()
        if bool(row["active"]) and int(idx) not in exclude
    ]
    if len(candidates) < k:
        raise RuntimeError(f"Only {len(candidates)} active non-excluded features available for k={k}")
    out = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < int(n_sets):
        sample = tuple(int(x) for x in rng.choice(candidates, size=k, replace=False).tolist())
        key = tuple(sorted(sample))
        if key in seen:
            continue
        seen.add(key)
        out.append(list(sample))
    return out


def _write_manifest(
    path: Path,
    *,
    feature_set: str,
    values: list[tuple[int, float]],
    task_spec: str,
    condition_name: str,
    is_random: bool,
    trigger_start_step: int,
    trigger_end_step: int | None = None,
    trigger_latch: int = 1,
    selection_strategy: str,
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
            "intervention_direction": "set_progress_features_to_calibrated_high_mean",
            "is_random_control": int(bool(is_random)),
            "trigger_mode": "always",
            "trigger_start_step": int(trigger_start_step),
            "trigger_latch": int(trigger_latch),
            "allowed_task_specs": task_spec,
        }
        if trigger_end_step is not None:
            row["trigger_end_step"] = int(trigger_end_step)
        if condition_name:
            row["condition_names"] = condition_name
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"

    stats = _class_stats_from_episode_features(args.episode_features_csv)
    submitted = _load_submitted(args.submitted_top_csv, stats)
    robust = _load_robust(args.robust_fdr_csv, stats)
    excluded = set(submitted) | set(robust)
    random_sets = _random_sets(
        stats=stats,
        exclude=excluded,
        k=20,
        n_sets=int(args.num_random_controls),
        seed=int(args.seed),
    )

    jobs: list[dict[str, str | int]] = []

    def add_family(
        *,
        family_name: str,
        features: list[int],
        alpha: float,
        task_spec: str = "goal:1",
        condition_name: str = "",
        n: int,
        trigger_start_step: int,
        trigger_end_step: int | None = None,
        trigger_latch: int = 1,
        random_controls: bool = True,
    ) -> None:
        alpha_name = str(alpha).replace(".", "p")
        cond_name = condition_name or "all"
        pulse_name = (
            f"start{trigger_start_step}"
            if trigger_end_step is None
            else f"pulse{trigger_start_step}_{trigger_end_step}_latch{trigger_latch}"
        )
        base = f"{family_name}_{task_spec.replace(':', '')}_{cond_name}_alpha{alpha_name}_{pulse_name}_n{n}"
        manifest = manifest_dir / f"{base}.csv"
        strategy = f"calibrated_positive_{family_name}"
        _write_manifest(
            manifest,
            feature_set=base,
            values=_values(features, stats, alpha),
            task_spec=task_spec,
            condition_name=condition_name,
            is_random=False,
            trigger_start_step=trigger_start_step,
            trigger_end_step=trigger_end_step,
            trigger_latch=trigger_latch,
            selection_strategy=strategy,
        )
        jobs.append(
            {
                "job_name": base,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_calibrated_positive_batch/{base}",
                "output_name": f"{base}.json",
            }
        )
        if not random_controls:
            return
        for i, feats in enumerate(random_sets):
            name = f"{base}_random{i:02d}"
            random_manifest = manifest_dir / f"{name}.csv"
            _write_manifest(
                random_manifest,
                feature_set=name,
                values=_values(feats, stats, alpha),
                task_spec=task_spec,
                condition_name=condition_name,
                is_random=True,
                trigger_start_step=trigger_start_step,
                trigger_end_step=trigger_end_step,
                trigger_latch=trigger_latch,
                selection_strategy=f"{strategy}_matched_random",
            )
            jobs.append(
                {
                    "job_name": name,
                    "manifest": str(random_manifest),
                    "num_rollouts": int(n),
                    "output_dir": f"results/eccv_calibrated_positive_batch/{name}",
                    "output_name": f"{name}.json",
                }
            )

    # Known less aggressive setting: verify with many controls.
    add_family(
        family_name="submitted20",
        features=submitted,
        alpha=1.0,
        condition_name="",
        n=160,
        trigger_start_step=20,
    )
    add_family(
        family_name="submitted20",
        features=submitted,
        alpha=1.0,
        condition_name="mild",
        n=128,
        trigger_start_step=20,
    )

    # Earlier intervention looked promising at n=64; now test specificity.
    add_family(
        family_name="submitted20",
        features=submitted,
        alpha=1.5,
        condition_name="",
        n=128,
        trigger_start_step=0,
    )
    add_family(
        family_name="submitted20",
        features=submitted,
        alpha=1.5,
        condition_name="mild",
        n=128,
        trigger_start_step=0,
    )

    # Short pulse: tries to preserve early steering without late oversteer.
    add_family(
        family_name="submitted20",
        features=submitted,
        alpha=1.5,
        condition_name="",
        n=128,
        trigger_start_step=0,
        trigger_end_step=40,
        trigger_latch=0,
    )

    # Robust cached-analysis features: if these work, claim training-split sparse
    # progress directions rather than relying on stale submitted feature IDs.
    add_family(
        family_name="robust20",
        features=robust,
        alpha=1.0,
        condition_name="",
        n=128,
        trigger_start_step=20,
    )
    add_family(
        family_name="robust20",
        features=robust,
        alpha=1.0,
        condition_name="mild",
        n=128,
        trigger_start_step=0,
    )

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    metadata = {
        "submitted_features": submitted,
        "robust_features": robust,
        "num_random_controls": int(args.num_random_controls),
        "num_jobs": len(jobs),
    }
    pd.Series(metadata, dtype=object).to_json(out_dir / "metadata.json", indent=2)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'}")
    print(f"Submitted features: {submitted}")
    print(f"Robust features: {robust}")


if __name__ == "__main__":
    main()
