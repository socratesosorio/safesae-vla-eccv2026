"""Evaluate runtime safety monitors and baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate safety monitor")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--output_dir", type=str, default="results/monitor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation_mode", type=str, default="")
    parser.add_argument("--split_mode", type=str, default="")
    parser.add_argument("--prediction_target", type=str, default="")
    parser.add_argument("--future_horizon", type=int, default=-1)
    parser.add_argument("--prefix_stride", type=int, default=-1)
    parser.add_argument("--min_prefix", type=int, default=-1)
    parser.add_argument("--target_category", type=str, default="")
    parser.add_argument("--task_eval_mode", type=str, default="")
    parser.add_argument("--task_eval_repeats", type=int, default=-1)
    parser.add_argument("--task_eval_test_groups", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "src.monitor.evaluate_monitor",
        "--sae_checkpoint",
        args.sae_checkpoint,
        "--layer",
        str(args.layer),
        "--data_dir",
        args.data_dir,
        "--sae_config",
        args.sae_config,
        "--eval_config",
        args.eval_config,
        "--rollout_config",
        args.rollout_config,
        "--output_dir",
        args.output_dir,
        "--seed",
        str(args.seed),
        "--future_horizon",
        str(args.future_horizon),
        "--prefix_stride",
        str(args.prefix_stride),
        "--min_prefix",
        str(args.min_prefix),
        "--task_eval_repeats",
        str(args.task_eval_repeats),
        "--task_eval_test_groups",
        str(args.task_eval_test_groups),
    ]

    for flag, value in (
        ("--evaluation_mode", args.evaluation_mode),
        ("--split_mode", args.split_mode),
        ("--prediction_target", args.prediction_target),
        ("--target_category", args.target_category),
        ("--task_eval_mode", args.task_eval_mode),
    ):
        if value:
            cmd.extend([flag, value])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
