"""Prepare broad suite/task sweep manifests for ECCV rebuttal triage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--random_manifest", type=str, default="logs/openvla_class_mean_intervention_prep/per_set_spatial1_n32/random20_high_class_mean_00.csv")
    p.add_argument("--out_dir", type=str, default="logs/eccv_rebuttal_suite_sweep_batch")
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--num_rollouts", type=int, default=32)
    return p.parse_args()


def _top_values(path: str, alpha: float) -> list[tuple[int, float]]:
    top = pd.read_csv(path).head(20)
    values = []
    for row in top.itertuples(index=False):
        low = float(row.mean_low_progress)
        high = float(row.mean_high_progress)
        values.append((int(row.feature_idx), low + float(alpha) * (high - low)))
    return values


def _random_values(path: str) -> list[tuple[int, float]]:
    df = pd.read_csv(path).sort_values(["rank", "feature_idx"], kind="stable")
    return [(int(row.feature_idx), float(row.feature_value)) for row in df.itertuples(index=False)]


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
                "selection_strategy": "suite_sweep_class_mean",
                "intervention_direction": "set_low_progress_toward_high_progress_mean",
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
    top_values = _top_values(args.top_features_csv, float(args.alpha))
    random_values = _random_values(args.random_manifest)

    jobs = []
    for suite in ["object", "goal", "long"]:
        for task_idx in range(4):
            for kind, values, is_random in [
                ("top20", top_values, False),
                ("random20_00", random_values, True),
            ]:
                name = f"{suite}{task_idx}_{kind}_alpha{str(args.alpha).replace('.', 'p')}_n{int(args.num_rollouts)}"
                manifest = manifest_dir / f"{name}.csv"
                _write_manifest(
                    manifest,
                    feature_set=name,
                    values=values,
                    task_spec=f"{suite}:{task_idx}",
                    is_random=is_random,
                )
                jobs.append(
                    {
                        "job_name": name,
                        "manifest": str(manifest),
                        "num_rollouts": int(args.num_rollouts),
                        "output_dir": f"results/eccv_rebuttal_suite_sweep_batch/{name}",
                        "output_name": f"{name}.json",
                    }
                )
    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'}")


if __name__ == "__main__":
    main()
