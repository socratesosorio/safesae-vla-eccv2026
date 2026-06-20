"""Run monitor evaluation across multiple SAE layers and aggregate outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monitor eval across multiple SAE layers")
    parser.add_argument("--layers", type=str, default="16,20,24")
    parser.add_argument("--checkpoint_template", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--output_root", type=str, required=True)
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
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def parse_layers(spec: str) -> list[int]:
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        raise ValueError("No layers specified")
    return layers


def eval_command(args: argparse.Namespace, layer: int, checkpoint_path: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.monitor.evaluate_monitor",
        "--sae_checkpoint",
        checkpoint_path,
        "--layer",
        str(layer),
        "--data_dir",
        args.data_dir,
        "--sae_config",
        args.sae_config,
        "--eval_config",
        args.eval_config,
        "--rollout_config",
        args.rollout_config,
        "--output_dir",
        str(output_dir),
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
    return cmd


def maybe_read_csv(path: Path, layer: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "layer", int(layer))
    return df


def aggregate_outputs(output_root: Path, layers: list[int]) -> None:
    artifacts = {
        "cross_layer_monitor_metrics.csv": "layer{layer}_monitor_metrics.csv",
        "cross_layer_monitor_metrics_by_split.csv": "layer{layer}_monitor_metrics_by_split.csv",
        "cross_layer_operating_points.csv": "layer{layer}_operating_points.csv",
        "cross_layer_operating_points_by_split.csv": "layer{layer}_operating_points_by_split.csv",
        "cross_layer_sae_feature_weights.csv": "layer{layer}_sae_feature_weights.csv",
        "cross_layer_sae_feature_weights_by_split.csv": "layer{layer}_sae_feature_weights_by_split.csv",
    }

    for out_name, pattern in artifacts.items():
        frames = []
        for layer in layers:
            layer_dir = output_root / f"layer{layer}"
            frames.append(maybe_read_csv(layer_dir / pattern.format(layer=layer), layer=layer))
        merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(
            not frame.empty for frame in frames
        ) else pd.DataFrame()
        merged.to_csv(output_root / out_name, index=False)

    monitor_summary = output_root / "cross_layer_monitor_metrics.csv"
    if monitor_summary.exists():
        df = pd.read_csv(monitor_summary)
        sae_df = df[df["method"] == "sae_lr"].copy() if "method" in df.columns else pd.DataFrame()
        sae_df.to_csv(output_root / "cross_layer_sae_lr_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    for layer in layers:
        checkpoint_path = args.checkpoint_template.format(layer=layer)
        layer_output_dir = output_root / f"layer{layer}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = layer_output_dir / f"layer{layer}_monitor_metrics.csv"
        if args.skip_existing and metrics_path.exists():
            continue

        subprocess.run(
            eval_command(args=args, layer=layer, checkpoint_path=checkpoint_path, output_dir=layer_output_dir),
            check=True,
        )

    aggregate_outputs(output_root=output_root, layers=layers)


if __name__ == "__main__":
    main()
