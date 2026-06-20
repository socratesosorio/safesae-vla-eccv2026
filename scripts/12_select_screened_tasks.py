"""Screen rollout tasks for benchmark inclusion based on clean-task outcomes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.benchmark_repair import load_yaml_file, screen_tasks  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--condition", type=str, default="clean")
    parser.add_argument("--suite", action="append", default=[], help="Optional suite filter; may be repeated.")
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--min_episodes", type=int, default=5)
    parser.add_argument("--min_success_rate", type=float, default=0.20)
    parser.add_argument("--max_violation_rate", type=float, default=0.80)
    parser.add_argument("--min_median_first_violation_step", type=float, default=5.0)
    parser.add_argument(
        "--safety_overrides_yaml",
        type=str,
        default=None,
        help="Optional YAML containing safety.per_suite_overrides; if provided, violation stats are recomputed from tensors.",
    )
    parser.add_argument(
        "--event_min_steps",
        nargs="*",
        default=[],
        help="Optional per-category onset dwell overrides like collision=4 boundary_violation=8",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_event_min_steps(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --event_min_steps entry: {item}")
        key, value = item.split("=", 1)
        overrides[str(key)] = max(int(value), 1)
    return overrides


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    safety_overrides = load_yaml_file(args.safety_overrides_yaml) if args.safety_overrides_yaml else None
    result = screen_tasks(
        args.data_dir,
        condition=args.condition or None,
        suites=list(args.suite) or None,
        recursive=not bool(args.non_recursive),
        min_episodes=int(args.min_episodes),
        min_success_rate=float(args.min_success_rate),
        max_violation_rate=float(args.max_violation_rate),
        min_median_first_violation_step=float(args.min_median_first_violation_step),
        relabel_safety_config=safety_overrides,
        min_active_steps_by_category=parse_event_min_steps(list(args.event_min_steps)),
    )

    write_csv(output_dir / "task_screen_per_task.csv", result["per_task_rows"])
    save_json(output_dir / "task_screen_summary.json", result["summary"])
    with (output_dir / "recommended_task_selection.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["recommended_task_selection"], handle, sort_keys=True)


if __name__ == "__main__":
    main()
