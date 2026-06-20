"""Calibrate per-suite safety thresholds from clean rollout statistics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.benchmark_repair import calibrate_safety_thresholds  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--condition", type=str, default="clean")
    parser.add_argument("--suite", action="append", default=[], help="Optional suite filter; may be repeated.")
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--boundary_lower_quantile", type=float, default=0.01)
    parser.add_argument("--boundary_upper_quantile", type=float, default=0.99)
    parser.add_argument("--boundary_margin", type=float, default=0.05)
    parser.add_argument("--collision_force_quantile", type=float, default=0.995)
    parser.add_argument("--collision_force_scale", type=float, default=1.05)
    parser.add_argument("--excessive_force_quantile", type=float, default=0.999)
    parser.add_argument("--excessive_force_scale", type=float, default=1.10)
    parser.add_argument("--speed_quantile", type=float, default=0.995)
    parser.add_argument("--speed_scale", type=float, default=1.05)
    parser.add_argument("--drop_velocity_quantile", type=float, default=0.995)
    parser.add_argument("--drop_velocity_scale", type=float, default=1.10)
    parser.add_argument("--min_episodes_per_suite", type=int, default=3)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    result = calibrate_safety_thresholds(
        args.data_dir,
        condition=args.condition or None,
        suites=list(args.suite) or None,
        recursive=not bool(args.non_recursive),
        success_only=bool(args.success_only),
        dt=float(args.dt),
        boundary_lower_quantile=float(args.boundary_lower_quantile),
        boundary_upper_quantile=float(args.boundary_upper_quantile),
        boundary_margin=float(args.boundary_margin),
        collision_force_quantile=float(args.collision_force_quantile),
        collision_force_scale=float(args.collision_force_scale),
        excessive_force_quantile=float(args.excessive_force_quantile),
        excessive_force_scale=float(args.excessive_force_scale),
        speed_quantile=float(args.speed_quantile),
        speed_scale=float(args.speed_scale),
        drop_velocity_quantile=float(args.drop_velocity_quantile),
        drop_velocity_scale=float(args.drop_velocity_scale),
        min_episodes_per_suite=int(args.min_episodes_per_suite),
    )

    write_csv(output_dir / "threshold_calibration_per_episode.csv", result["per_episode_rows"])
    write_csv(output_dir / "threshold_calibration_per_suite.csv", result["per_suite_rows"])
    save_json(output_dir / "threshold_calibration_summary.json", result["summary"])
    with (output_dir / "recommended_safety_overrides.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["recommended_config"], handle, sort_keys=True)


if __name__ == "__main__":
    main()
