"""Vast.ai helper to bootstrap environment, train SAE, and sync checkpoints."""

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
    parser = argparse.ArgumentParser(description="Bootstrap and run SAE training on Vast.ai")
    parser.add_argument("--vast_config", type=str, default="configs/vast_config.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--backend", type=str, default="manual", choices=["manual", "saelens"])
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.vast_config).get("vast", {})

    data_dir = str(cfg.get("data_dir", "")).strip() or "/workspace/data/rollouts"
    checkpoint_dir = str(cfg.get("checkpoint_dir", "")).strip() or "/workspace/checkpoints"

    subprocess.run(
        [
            "python",
            "scripts/run_sae_training.py",
            "--provider",
            "vast",
            "--backend",
            args.backend,
            "--layer",
            str(args.layer),
            "--config",
            args.sae_config,
            "--vast_config",
            args.vast_config,
            "--data_dir",
            data_dir,
            "--checkpoint_dir",
            checkpoint_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
