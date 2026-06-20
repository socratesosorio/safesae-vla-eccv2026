"""Merge screened task selection and calibrated safety overrides into a benchmark config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.benchmark_repair import build_repaired_benchmark_config, load_yaml_file  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_config", type=str, default="configs/rollout_benchmark.yaml")
    parser.add_argument("--task_selection_yaml", type=str, required=True)
    parser.add_argument("--safety_overrides_yaml", type=str, required=True)
    parser.add_argument("--output_config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    task_selection_cfg = load_yaml_file(args.task_selection_yaml)
    safety_cfg = load_yaml_file(args.safety_overrides_yaml)
    merged = build_repaired_benchmark_config(
        base_cfg,
        task_selection_config=task_selection_cfg,
        safety_overrides_config=safety_cfg,
    )

    out_path = Path(args.output_config)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=True)
    save_json(
        out_path.with_suffix(".summary.json"),
        {
            "base_config": str(Path(args.base_config)),
            "task_selection_yaml": str(Path(args.task_selection_yaml)),
            "safety_overrides_yaml": str(Path(args.safety_overrides_yaml)),
            "output_config": str(out_path),
            "total_rollouts": int(merged.get("collection", {}).get("total_rollouts", 0)),
            "per_suite": merged.get("collection", {}).get("per_suite", {}),
        },
    )


if __name__ == "__main__":
    main()
