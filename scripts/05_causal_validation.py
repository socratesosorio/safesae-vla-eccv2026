"""Run two-phase causal validation: activation patching + simulator validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal validation pipeline")
    parser.add_argument("--model", type=str, default="openvla", choices=["openvla", "pi0"])
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--ranked_features", type=str, default="")
    parser.add_argument("--feature_manifest_csv", type=str, default="")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--data_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--output_dir", type=str, default="results/causal")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--scale", type=float, default=float("nan"))
    parser.add_argument("--num_rollouts", type=int, default=-1)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--random_controls_csv", type=str, default="")
    parser.add_argument("--target_category", type=str, default="")
    parser.add_argument("--hazard_category", type=str, default="")
    parser.add_argument("--condition_group", type=str, default="")
    parser.add_argument("--condition_names", type=str, default="")
    parser.add_argument("--allowed_suites", type=str, default="")
    parser.add_argument("--allowed_task_specs", type=str, default="")
    parser.add_argument("--trigger_mode", type=str, default="")
    parser.add_argument("--trigger_threshold", type=float, default=float("nan"))
    parser.add_argument("--trigger_start_step", type=int, default=-1)
    parser.add_argument("--trigger_end_step", type=int, default=-1)
    parser.add_argument("--trigger_latch", type=int, default=-1)
    parser.add_argument("--run_modal", action="store_true")
    parser.add_argument("--test", action="store_true", help="Skip --detach for quick testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not str(args.ranked_features).strip() and not str(args.feature_manifest_csv).strip():
        raise ValueError("Provide --ranked_features or --feature_manifest_csv")

    rollout_config = args.rollout_config
    if args.model == "pi0" and rollout_config == "configs/rollout.yaml":
        rollout_config = "configs/rollout_pi0.yaml"
    rollout_cfg = load_yaml(rollout_config)
    eval_cfg = load_yaml(args.eval_config)
    sim_cfg = eval_cfg.get("simulator_validation", {})
    top_k = int(args.top_k if args.top_k > 0 else sim_cfg.get("top_k_features", 5))
    scale = float(args.scale if args.scale == args.scale else sim_cfg.get("scale_factor", 0.0))
    num_rollouts = int(args.num_rollouts if args.num_rollouts > 0 else sim_cfg.get("num_clamped_rollouts", 100))
    acfg = rollout_cfg.get("activation_caching", {})
    if args.layer >= 0:
        layer = int(args.layer)
    elif "layer" in acfg:
        layer = int(acfg.get("layer", 11 if args.model == "pi0" else 20))
    else:
        default_layers = [9, 11, 14] if args.model == "pi0" else [16, 20, 24]
        layers = [int(x) for x in acfg.get("layers", default_layers)]
        preferred = 11 if args.model == "pi0" else 20
        layer = preferred if preferred in layers else int(layers[0])
    output_name = str(args.output_name).strip() or (
        "pi0_clamping_results.json" if args.model == "pi0" else "clamping_results.json"
    )

    top_features: list[int] = []
    if str(args.ranked_features).strip():
        ranked = pd.read_csv(args.ranked_features)
        top_features = ranked["feature_idx"].astype(int).head(top_k).tolist()

    # Local simulator validation entrypoint.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.causal_validation",
            "--rollout_config",
            rollout_config,
            "--sae_checkpoint",
            args.sae_checkpoint,
            "--sae_config",
            args.sae_config,
            *(["--features", args.ranked_features] if str(args.ranked_features).strip() else []),
            *(
                ["--feature_manifest_csv", args.feature_manifest_csv]
                if str(args.feature_manifest_csv).strip()
                else []
            ),
            "--top_k",
            str(top_k),
            "--scale",
            str(scale),
            "--num_rollouts",
            str(num_rollouts),
            "--layer",
            str(layer),
            "--output_dir",
            args.output_dir,
            "--output_name",
            output_name,
            *(
                ["--random_controls_csv", args.random_controls_csv]
                if str(args.random_controls_csv).strip()
                else []
            ),
            *(["--target_category", args.target_category] if str(args.target_category).strip() else []),
            *(["--hazard_category", args.hazard_category] if str(args.hazard_category).strip() else []),
            *(["--condition_group", args.condition_group] if str(args.condition_group).strip() else []),
            *(["--condition_names", args.condition_names] if str(args.condition_names).strip() else []),
            *(["--allowed_suites", args.allowed_suites] if str(args.allowed_suites).strip() else []),
            *(["--allowed_task_specs", args.allowed_task_specs] if str(args.allowed_task_specs).strip() else []),
            *(["--trigger_mode", args.trigger_mode] if str(args.trigger_mode).strip() else []),
            *(["--trigger_threshold", str(args.trigger_threshold)] if args.trigger_threshold == args.trigger_threshold else []),
            *(["--trigger_start_step", str(args.trigger_start_step)] if args.trigger_start_step >= 0 else []),
            *(["--trigger_end_step", str(args.trigger_end_step)] if args.trigger_end_step >= 0 else []),
            *(["--trigger_latch", str(int(args.trigger_latch))] if args.trigger_latch >= 0 else []),
        ],
        check=True,
    )

    if args.run_modal:
        if not top_features:
            raise ValueError("--run_modal requires --ranked_features so features can be serialized")
        features_arg = ",".join(str(x) for x in top_features)
        entry = "validate" if args.model == "openvla" else "validate_pi0"
        cmd = ["modal", "run"]
        if not args.test:
            cmd.append("--detach")
        cmd += [
            f"modal_app.py::{entry}",
            "--sae-path", args.sae_checkpoint,
            "--features", features_arg,
            "--rollout-config-path", rollout_config,
            "--eval-config-path", args.eval_config,
        ]
        subprocess.run(cmd, check=True)

    out_json = Path(args.output_dir) / output_name
    print(f"Causal validation complete: {out_json}")


if __name__ == "__main__":
    main()
