"""Generate all figures/tables for paper integration."""

from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SafeSAE-VLA figures")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--paper_dir", type=str, default="paper")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--sae_checkpoint", type=str, default="")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--max_tsne_episodes", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "python",
        "scripts/generate_figures.py",
        "--results_dir",
        args.results_dir,
        "--output_dir",
        args.output_dir,
        "--paper_dir",
        args.paper_dir,
        "--layer",
        str(args.layer),
        "--sae_config",
        args.sae_config,
        "--max_tsne_episodes",
        str(args.max_tsne_episodes),
    ]
    if args.data_dir:
        cmd.extend(["--data_dir", args.data_dir])
    if args.sae_checkpoint:
        cmd.extend(["--sae_checkpoint", args.sae_checkpoint])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
