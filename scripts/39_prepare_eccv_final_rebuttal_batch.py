"""Prepare final high-EV Athena jobs for ECCV rebuttal triage.

These jobs focus on the only currently clean closed-loop signal: goal:1
top-20 class-mean setting reduced violations while a matched random-20 control
did not. The batch strengthens or falsifies that task-local claim with larger
n, multiple random controls, and small sparsity/direction checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--top_features_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv",
    )
    p.add_argument(
        "--random_manifest_dir",
        type=str,
        default="logs/openvla_class_mean_intervention_prep/per_set_spatial1_n32",
    )
    p.add_argument("--out_dir", type=str, default="logs/eccv_final_rebuttal_batch")
    return p.parse_args()


def _class_value(top: pd.DataFrame, feature_idx: int, alpha: float) -> float:
    row = top.loc[int(feature_idx)]
    low = float(row["mean_low_progress"])
    high = float(row["mean_high_progress"])
    return low + float(alpha) * (high - low)


def _top_values(top: pd.DataFrame, alpha: float, k: int = 20) -> list[tuple[int, float]]:
    return [(int(f), _class_value(top, int(f), alpha)) for f in top.index.tolist()[:k]]


def _top_low_values(top: pd.DataFrame, k: int = 20) -> list[tuple[int, float]]:
    return [(int(f), float(top.loc[int(f), "mean_low_progress"])) for f in top.index.tolist()[:k]]


def _top_zero_values(top: pd.DataFrame, k: int = 20) -> list[tuple[int, float]]:
    return [(int(f), 0.0) for f in top.index.tolist()[:k]]


def _random_values(manifest_dir: Path, idx: int, *, zero: bool = False) -> list[tuple[int, float]]:
    path = manifest_dir / f"random20_high_class_mean_{idx:02d}.csv"
    df = pd.read_csv(path).sort_values(["rank", "feature_idx"], kind="stable")
    if zero:
        return [(int(row.feature_idx), 0.0) for row in df.itertuples(index=False)]
    return [(int(row.feature_idx), float(row.feature_value)) for row in df.itertuples(index=False)]


def _write_manifest(
    path: Path,
    *,
    feature_set: str,
    values: list[tuple[int, float]],
    task_spec: str,
    is_random: bool,
    trigger_start_step: int = 20,
    trigger_mode: str = "always",
    trigger_threshold: float | None = None,
    direction: str = "set_low_progress_toward_high_progress_mean",
) -> None:
    rows = []
    for rank, (feature_idx, feature_value) in enumerate(values):
        row = {
            "feature_set": feature_set,
            "rank": int(rank),
            "feature_idx": int(feature_idx),
            "feature_value": float(feature_value),
            "default_scale": 1.0,
            "selection_strategy": "eccv_final_rebuttal_high_ev",
            "intervention_direction": direction,
            "is_random_control": int(bool(is_random)),
            "trigger_mode": trigger_mode,
            "trigger_start_step": int(trigger_start_step),
            "allowed_task_specs": task_spec,
        }
        if trigger_threshold is not None:
            row["trigger_threshold"] = float(trigger_threshold)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    top = pd.read_csv(args.top_features_csv).head(20).set_index("feature_idx")
    random_manifest_dir = Path(args.random_manifest_dir)

    top20_1p0 = _top_values(top, 1.0)
    top20_1p5 = _top_values(top, 1.5)
    top20_2p0 = _top_values(top, 2.0)
    top20_low = _top_low_values(top)
    top20_zero = _top_zero_values(top)
    random_high = {i: _random_values(random_manifest_dir, i) for i in range(3)}
    random_zero = {i: _random_values(random_manifest_dir, i, zero=True) for i in range(3)}

    jobs: list[dict[str, str | int]] = []

    def add(
        name: str,
        *,
        values: list[tuple[int, float]],
        task_spec: str,
        n: int,
        is_random: bool = False,
        trigger_start_step: int = 20,
        trigger_mode: str = "always",
        trigger_threshold: float | None = None,
        direction: str = "set_low_progress_toward_high_progress_mean",
    ) -> None:
        manifest = manifest_dir / f"{name}.csv"
        _write_manifest(
            manifest,
            feature_set=name,
            values=values,
            task_spec=task_spec,
            is_random=is_random,
            trigger_start_step=trigger_start_step,
            trigger_mode=trigger_mode,
            trigger_threshold=trigger_threshold,
            direction=direction,
        )
        jobs.append(
            {
                "job_name": name,
                "manifest": str(manifest),
                "num_rollouts": int(n),
                "output_dir": f"results/eccv_final_rebuttal_batch/{name}",
                "output_name": f"{name}.json",
            }
        )

    # Confirmatory replication of the clean goal:1 result.
    add("goal1_top20_alpha1p5_n128", values=top20_1p5, task_spec="goal:1", n=128)
    for i in range(3):
        add(
            f"goal1_random20_{i:02d}_highmean_n128",
            values=random_high[i],
            task_spec="goal:1",
            n=128,
            is_random=True,
        )

    # Dose and sparsity checks on the same task-local setting.
    add("goal1_top20_alpha1p0_n96", values=top20_1p0, task_spec="goal:1", n=96)
    add("goal1_top20_alpha2p0_n96", values=top20_2p0, task_spec="goal:1", n=96)
    add("goal1_top5_alpha1p5_n96", values=top20_1p5[:5], task_spec="goal:1", n=96)
    add("goal1_top10_alpha1p5_n96", values=top20_1p5[:10], task_spec="goal:1", n=96)

    # Direction controls. Useful if high-mean helps and low/zero does not, or if
    # suppression reliably perturbs success in the opposite direction.
    add(
        "goal1_top20_lowmean_n96",
        values=top20_low,
        task_spec="goal:1",
        n=96,
        direction="set_progress_features_to_low_progress_mean",
    )
    add(
        "goal1_top20_zero_n96",
        values=top20_zero,
        task_spec="goal:1",
        n=96,
        direction="zero_progress_features",
    )
    for i in range(3):
        add(
            f"goal1_random20_{i:02d}_zero_n96",
            values=random_zero[i],
            task_spec="goal:1",
            n=96,
            is_random=True,
            direction="zero_random_active_features",
        )

    # Timing/trigger checks. Include only if they preserve the goal:1 pattern.
    for start_step in [0, 40, 80]:
        add(
            f"goal1_top20_alpha1p5_start{start_step}_n64",
            values=top20_1p5,
            task_spec="goal:1",
            n=64,
            trigger_start_step=start_step,
        )
    add(
        "goal1_top20_alpha1p5_speedtrigger08_n64",
        values=top20_1p5,
        task_spec="goal:1",
        n=64,
        trigger_mode="eef_speed_ratio",
        trigger_threshold=0.8,
    )
    add(
        "goal1_top20_alpha1p5_boundarytrigger_n64",
        values=top20_1p5,
        task_spec="goal:1",
        n=64,
        trigger_mode="boundary_margin",
        trigger_threshold=0.02,
    )

    # Replicate the two success-improvement hints, each with all random controls.
    for task_spec in ["goal:3", "long:0"]:
        safe = task_spec.replace(":", "")
        add(f"{safe}_top20_alpha1p5_n96", values=top20_1p5, task_spec=task_spec, n=96)
        for i in range(3):
            add(
                f"{safe}_random20_{i:02d}_highmean_n96",
                values=random_high[i],
                task_spec=task_spec,
                n=96,
                is_random=True,
            )

    pd.DataFrame(jobs).to_csv(out_dir / "jobs.tsv", sep="\t", index=False)
    print(f"Wrote {len(jobs)} jobs to {out_dir / 'jobs.tsv'}")


if __name__ == "__main__":
    main()
