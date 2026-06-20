"""Prepare closed-loop negative-intervention jobs for ECCV rebuttal triage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--random_manifest_dir", type=str, default="logs/openvla_class_mean_intervention_prep/per_set_spatial1_n32")
    p.add_argument("--out_dir", type=str, default="logs/eccv_rebuttal_ablation_batch")
    return p.parse_args()


def _top_values(path: str, mode: str) -> list[tuple[int, float]]:
    top = pd.read_csv(path).head(20)
    values = []
    for row in top.itertuples(index=False):
        if mode == "lowmean":
            value = float(row.mean_low_progress)
        elif mode == "zero":
            value = 0.0
        else:
            raise ValueError(mode)
        values.append((int(row.feature_idx), value))
    return values


def _random_zero_values(manifest_dir: str, idx: int) -> list[tuple[int, float]]:
    path = Path(manifest_dir) / f"random20_high_class_mean_{idx:02d}.csv"
    df = pd.read_csv(path).sort_values(["rank", "feature_idx"], kind="stable")
    return [(int(row.feature_idx), 0.0) for row in df.itertuples(index=False)]


def _write_manifest(path: Path, *, feature_set: str, values: list[tuple[int, float]], task_spec: str, is_random: bool) -> None:
    rows = []
    for rank, (feature_idx, feature_value) in enumerate(values):
        rows.append(
            {
                "feature_set": feature_set,
                "rank": rank,
                "feature_idx": int(feature_idx),
                "feature_value": float(feature_value),
                "default_scale": 1.0,
                "selection_strategy": "negative_intervention",
                "intervention_direction": "suppress_progress_feature_value",
                "is_random_control": int(bool(is_random)),
                "trigger_mode": "always",
                "trigger_start_step": 20,
                "allowed_task_specs": task_spec,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    top_low = _top_values(args.top_features_csv, "lowmean")
    top_zero = _top_values(args.top_features_csv, "zero")
    random_zero = {i: _random_zero_values(args.random_manifest_dir, i) for i in range(3)}

    jobs = []

    def add(name: str, *, values: list[tuple[int, float]], task_spec: str, n: int, is_random: bool = False) -> None:
        manifest = manifest_dir / f"{name}.csv"
        _write_manifest(manifest, feature_set=name, values=values, task_spec=task_spec, is_random=is_random)
        jobs.append(
            {
                "job_name": name,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_rebuttal_ablation_batch/{name}",
                "output_name": f"{name}.json",
            }
        )

    # Direct negative controls on the same high-success spatial task.
    add("spatial1_top20_lowmean_n64", values=top_low, task_spec="spatial:1", n=64)
    add("spatial1_top20_zero_n64", values=top_zero, task_spec="spatial:1", n=64)
    for i in range(3):
        add(f"spatial1_random20_{i:02d}_zero_n64", values=random_zero[i], task_spec="spatial:1", n=64, is_random=True)

    # Sparse concentration of degradation: if top-5/top-10 already hurt, that is strong.
    add("spatial1_top5_zero_n48", values=top_zero[:5], task_spec="spatial:1", n=48)
    add("spatial1_top10_zero_n48", values=top_zero[:10], task_spec="spatial:1", n=48)

    # Adjacent spatial tasks. These are exploratory; include only if top-specific and clean.
    for task_idx in [0, 2, 3]:
        add(f"spatial{task_idx}_top20_zero_n40", values=top_zero, task_spec=f"spatial:{task_idx}", n=40)
        add(f"spatial{task_idx}_random20_00_zero_n40", values=random_zero[0], task_spec=f"spatial:{task_idx}", n=40, is_random=True)

    # One broader high-level sanity check on goal/long tasks that had room for success effects.
    for task_spec in ["goal:5", "long:1", "long:6"]:
        safe_name = task_spec.replace(":", "")
        add(f"{safe_name}_top20_zero_n32", values=top_zero, task_spec=task_spec, n=32)
        add(f"{safe_name}_random20_00_zero_n32", values=random_zero[0], task_spec=task_spec, n=32, is_random=True)

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'}")


if __name__ == "__main__":
    main()
