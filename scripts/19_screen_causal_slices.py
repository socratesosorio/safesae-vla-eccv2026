"""Screen recoverable hazard-targeted task-condition cells for causal validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.benchmark_repair import SAFETY_CATEGORIES, screen_causal_slices
from src.utils.runtime import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen recoverable causal validation slices")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=",".join(SAFETY_CATEGORIES))
    parser.add_argument("--condition_group", type=str, default="hazard_targeted")
    parser.add_argument("--min_episodes", type=int, default=4)
    parser.add_argument("--min_success_rate", type=float, default=0.05)
    parser.add_argument("--min_clean_success_rate", type=float, default=0.0)
    parser.add_argument("--min_target_rate", type=float, default=0.10)
    parser.add_argument("--max_target_rate", type=float, default=0.90)
    parser.add_argument("--max_any_violation_rate", type=float, default=1.0)
    parser.add_argument("--max_cells_per_category", type=int, default=6)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = [part.strip() for part in str(args.categories).split(",") if part.strip()]
    out_dir = Path(ensure_dir(args.output_dir))
    result = screen_causal_slices(
        args.data_dir,
        recursive=bool(args.recursive),
        categories=categories,
        condition_group=str(args.condition_group),
        min_episodes=int(args.min_episodes),
        min_success_rate=float(args.min_success_rate),
        min_clean_success_rate=float(args.min_clean_success_rate),
        min_target_rate=float(args.min_target_rate),
        max_target_rate=float(args.max_target_rate),
        max_any_violation_rate=float(args.max_any_violation_rate),
        max_cells_per_category=int(args.max_cells_per_category),
    )

    pd.DataFrame(result["per_cell_rows"]).to_csv(out_dir / "causal_slice_screen_per_cell.csv", index=False)
    pd.DataFrame(result["recommended_rows"]).to_csv(out_dir / "causal_slice_screen_recommended.csv", index=False)
    save_json(out_dir / "causal_slice_screen_summary.json", result)


if __name__ == "__main__":
    main()
