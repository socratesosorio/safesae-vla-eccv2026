"""Collect full rollout dataset locally or via Modal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rollout dataset")
    parser.add_argument("--mode", type=str, default="modal", choices=["modal", "local"])
    parser.add_argument("--model", type=str, default="openvla", choices=["openvla", "pi0"])
    parser.add_argument("--config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=None, help="Start rollout index (for multi-GPU local runs)")
    parser.add_argument("--end_idx", type=int, default=None, help="End rollout index (for multi-GPU local runs)")
    parser.add_argument("--test", action="store_true", help="Skip --detach for quick testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if args.model == "pi0" and config_path == "configs/rollout.yaml":
        config_path = "configs/rollout_pi0.yaml"
    if args.mode == "local":
        cfg = load_yaml(config_path)
        if args.model == "openvla":
            from src.data.rollout_collector import RolloutCollector

            collector = RolloutCollector(cfg)
        else:
            from src.data.pi0_rollout_collector import Pi0RolloutCollector

            collector = Pi0RolloutCollector(cfg)
        if args.start_idx is not None and args.end_idx is not None:
            collector.collect_range(args.start_idx, args.end_idx, args.output_dir)
            print(f"Local {args.model} rollout collection complete (range {args.start_idx}-{args.end_idx}): {args.output_dir}")
        else:
            collector.collect_all(args.output_dir)
            print(f"Local {args.model} rollout collection complete: {args.output_dir}")
        return

    entry = "collect" if args.model == "openvla" else "collect_pi0"
    cmd = ["modal", "run"]
    if not args.test:
        cmd.append("--detach")
    cmd += [f"modal_app.py::{entry}", "--config-path", config_path, "--num-workers", str(args.workers)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
