"""Audit success signals and reward summaries from collected rollouts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.benchmark_repair import audit_success_signals  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--suite", action="append", default=[], help="Optional suite filter; may be repeated.")
    parser.add_argument("--non_recursive", action="store_true")
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
    result = audit_success_signals(
        args.data_dir,
        condition=args.condition,
        suites=list(args.suite) or None,
        recursive=not bool(args.non_recursive),
    )
    write_csv(output_dir / "success_audit_per_episode.csv", result["per_episode_rows"])
    write_csv(output_dir / "success_audit_per_task.csv", result["per_task_rows"])
    save_json(output_dir / "success_audit_summary.json", result["summary"])


if __name__ == "__main__":
    main()
