"""Run full analysis stack: differential stats, monitor eval, optional causal validation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SafeSAE-VLA analysis pipeline")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_causal", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=500)
    parser.add_argument("--generate_figures", action="store_true")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--paper_dir", type=str, default="paper")
    parser.add_argument("--max_tsne_episodes", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    diff_out = out / "differential"
    mon_out = out / "monitor"
    causal_out = out / "causal"

    subprocess.run(
        [
            "python",
            "-m",
            "src.analysis.differential_activation",
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
            "--output_dir",
            str(diff_out),
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
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
            str(mon_out),
        ],
        check=True,
    )

    if args.run_causal:
        feature_csv = diff_out / f"layer{args.layer}_overall.csv"
        subprocess.run(
            [
                "python",
                "-m",
                "src.analysis.causal_validation",
                "--features",
                str(feature_csv),
                "--sae_checkpoint",
                args.sae_checkpoint,
                "--layer",
                str(args.layer),
                "--rollout_config",
                args.rollout_config,
                "--sae_config",
                args.sae_config,
                "--eval_config",
                args.eval_config,
                "--num_rollouts",
                str(args.num_rollouts),
                "--output_dir",
                str(causal_out),
            ],
            check=True,
        )

    if args.generate_figures:
        subprocess.run(
            [
                "python",
                "scripts/generate_figures.py",
                "--results_dir",
                str(out),
                "--output_dir",
                args.figures_dir,
                "--paper_dir",
                args.paper_dir,
                "--layer",
                str(args.layer),
                "--data_dir",
                args.data_dir,
                "--sae_checkpoint",
                args.sae_checkpoint,
                "--sae_config",
                args.sae_config,
                "--max_tsne_episodes",
                str(args.max_tsne_episodes),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
