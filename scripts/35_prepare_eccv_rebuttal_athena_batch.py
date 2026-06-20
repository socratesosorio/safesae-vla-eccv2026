"""Prepare Athena job manifests for last-mile ECCV rebuttal experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--random_manifest_dir", type=str, default="logs/openvla_class_mean_intervention_prep/per_set_spatial1_n32")
    p.add_argument("--out_dir", type=str, default="logs/eccv_rebuttal_athena_batch")
    return p.parse_args()


def _class_value(top: pd.DataFrame, feature_idx: int, alpha: float) -> float:
    row = top.loc[int(feature_idx)]
    low = float(row["mean_low_progress"])
    high = float(row["mean_high_progress"])
    return low + float(alpha) * (high - low)


def _make_manifest(
    *,
    feature_values: list[tuple[int, float]],
    feature_set: str,
    task_spec: str,
    trigger_start_step: int,
    trigger_mode: str = "always",
    trigger_threshold: float | None = None,
    is_random_control: bool = False,
    out_path: Path,
) -> None:
    rows = []
    for rank, (feature_idx, feature_value) in enumerate(feature_values):
        row = {
            "feature_set": feature_set,
            "rank": rank,
            "feature_idx": int(feature_idx),
            "feature_value": float(feature_value),
            "default_scale": 1.0,
            "selection_strategy": "class_mean_set",
            "intervention_direction": "set_low_progress_toward_high_progress_mean",
            "is_random_control": int(bool(is_random_control)),
            "trigger_mode": trigger_mode,
            "trigger_start_step": int(trigger_start_step),
            "allowed_task_specs": task_spec,
        }
        if trigger_threshold is not None:
            row["trigger_threshold"] = float(trigger_threshold)
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    top = pd.read_csv(args.top_features_csv).head(20).set_index("feature_idx")
    top_features = [int(x) for x in top.index.tolist()]
    random_manifest_dir = Path(args.random_manifest_dir)
    random_sets: dict[str, list[tuple[int, float]]] = {}
    for i in range(3):
        path = random_manifest_dir / f"random20_high_class_mean_{i:02d}.csv"
        rdf = pd.read_csv(path).sort_values(["rank", "feature_idx"], kind="stable")
        random_sets[f"random20_{i:02d}"] = [
            (int(row.feature_idx), float(row.feature_value)) for row in rdf.itertuples(index=False)
        ]

    jobs: list[dict[str, str | int | float]] = []

    def add_job(
        name: str,
        *,
        feature_values: list[tuple[int, float]],
        task_spec: str,
        n: int,
        trigger_start_step: int = 20,
        trigger_mode: str = "always",
        trigger_threshold: float | None = None,
        is_random_control: bool = False,
    ) -> None:
        manifest = manifest_dir / f"{name}.csv"
        _make_manifest(
            feature_values=feature_values,
            feature_set=name,
            task_spec=task_spec,
            trigger_start_step=trigger_start_step,
            trigger_mode=trigger_mode,
            trigger_threshold=trigger_threshold,
            is_random_control=is_random_control,
            out_path=manifest,
        )
        jobs.append(
            {
                "job_name": name,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_rebuttal_athena_batch/{name}",
                "output_name": f"{name}.json",
            }
        )

    # Highest-value confirmatory runs: more power on the focused spatial:1 result.
    top20_1p0 = [(f, _class_value(top, f, 1.0)) for f in top_features]
    top20_1p5 = [(f, _class_value(top, f, 1.5)) for f in top_features]
    top20_2p0 = [(f, _class_value(top, f, 2.0)) for f in top_features]
    add_job("spatial1_top20_alpha1p0_n96", feature_values=top20_1p0, task_spec="spatial:1", n=96)
    add_job("spatial1_top20_alpha1p5_n96", feature_values=top20_1p5, task_spec="spatial:1", n=96)
    for i in range(3):
        add_job(
            f"spatial1_random20_{i:02d}_highmean_n64",
            feature_values=random_sets[f"random20_{i:02d}"],
            task_spec="spatial:1",
            n=64,
            is_random_control=True,
        )

    # Sparse concentration: can we get the same safety effect from fewer named directions?
    add_job("spatial1_top5_alpha1p5_n48", feature_values=top20_1p5[:5], task_spec="spatial:1", n=48)
    add_job("spatial1_top10_alpha1p5_n48", feature_values=top20_1p5[:10], task_spec="spatial:1", n=48)

    # Adjacent task probes. Include one matched random control per task; only use if clean.
    for task_idx in [0, 2, 3]:
        add_job(f"spatial{task_idx}_top20_alpha1p5_n48", feature_values=top20_1p5, task_spec=f"spatial:{task_idx}", n=48)
        add_job(
            f"spatial{task_idx}_random20_00_highmean_n48",
            feature_values=random_sets["random20_00"],
            task_spec=f"spatial:{task_idx}",
            n=48,
            is_random_control=True,
        )

    # Timing controls: useful if the safety effect is not just an arbitrary constant perturbation.
    for start_step in [0, 10, 40, 80]:
        add_job(
            f"spatial1_top20_alpha1p5_start{start_step}_n48",
            feature_values=top20_1p5,
            task_spec="spatial:1",
            n=48,
            trigger_start_step=start_step,
        )

    # Triggered policies. These are exploratory and should be included only if they are clean.
    add_job(
        "spatial1_top20_alpha1p5_speedtrigger08_n48",
        feature_values=top20_1p5,
        task_spec="spatial:1",
        n=48,
        trigger_mode="eef_speed_ratio",
        trigger_threshold=0.8,
    )
    add_job(
        "spatial1_top20_alpha1p5_boundarytrigger_n48",
        feature_values=top20_1p5,
        task_spec="spatial:1",
        n=48,
        trigger_mode="boundary_margin",
        trigger_threshold=0.02,
    )

    # Saturation check. If it preserves success and improves violations, it strengthens the dose story.
    add_job("spatial1_top20_alpha2p0_n48", feature_values=top20_2p0, task_spec="spatial:1", n=48)

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'}")


if __name__ == "__main__":
    main()
