"""Run or prepare direct OpenVLA class-mean feature-setting interventions.

The experiment sets top progress SAE features to their high-progress class
means during OpenVLA inference. This avoids the inactive-feature failure mode
of multiplicative scaling and measures policy-output / closed-loop effects via
the existing paired causal-validation runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--rollout_config", type=str, default="configs/rollout_semantic_audit.yaml")
    p.add_argument("--sae_checkpoint", type=str, default="results/athena_pilot/checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    p.add_argument("--output_dir", type=str, default="results/openvla_class_mean_intervention")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--random_trials", type=int, default=3)
    p.add_argument("--num_rollouts", type=int, default=60)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--trigger_start_step", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Only write manifest and command metadata.")
    return p.parse_args()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _write_manifest(args: argparse.Namespace, out_dir: Path) -> Path:
    df = pd.read_csv(args.episode_features_csv)
    if "label" not in df.columns:
        raise ValueError(f"{args.episode_features_csv} must contain a label column with high-progress label=1")
    top = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(int(args.top_k)).tolist()
    top_name = f"top{int(args.top_k)}_high_class_mean"
    random_name = f"random{int(args.top_k)}_high_class_mean"
    feat_cols = _feature_cols(df)
    active = [int(c[1:]) for c in feat_cols if (df[c] > 0).any()]
    top_set = set(top)
    active_not_top = [idx for idx in active if idx not in top_set]
    high = df[df["label"].astype(int) == 1]
    if high.empty:
        raise ValueError("No high-progress rows found for class-mean patch values")

    rng = np.random.default_rng(int(args.seed))
    rows = []
    for rank, feat_idx in enumerate(top):
        col = f"f{feat_idx}"
        rows.append(
            {
                "feature_set": top_name,
                "rank": rank,
                "feature_idx": int(feat_idx),
                "feature_value": float(high[col].mean()),
                "default_scale": 1.0,
                "selection_strategy": "class_mean_set",
                "intervention_direction": "set_low_progress_toward_high_progress_mean",
                "is_random_control": 0,
                "trigger_mode": "always",
                "trigger_start_step": int(args.trigger_start_step),
            }
        )
    for trial in range(int(args.random_trials)):
        if len(active_not_top) < len(top):
            sampled = rng.choice(active, size=len(top), replace=False)
        else:
            sampled = rng.choice(active_not_top, size=len(top), replace=False)
        for rank, feat_idx in enumerate(sampled):
            col = f"f{int(feat_idx)}"
            rows.append(
                {
                "feature_set": f"{random_name}_{trial:02d}",
                    "rank": rank,
                    "feature_idx": int(feat_idx),
                    "feature_value": float(high[col].mean()),
                    "default_scale": 1.0,
                    "selection_strategy": "random_class_mean_set",
                    "intervention_direction": "random_set_to_high_progress_mean",
                    "is_random_control": 1,
                    "trigger_mode": "always",
                    "trigger_start_step": int(args.trigger_start_step),
                }
            )
    manifest = out_dir / "class_mean_intervention_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(args, out_dir)
    command = {
        "manifest": str(manifest),
        "rollout_config": args.rollout_config,
        "sae_checkpoint": args.sae_checkpoint,
        "num_rollouts": int(args.num_rollouts),
        "layer": int(args.layer),
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "class_mean_intervention_command.json").write_text(json.dumps(command, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(command, indent=2, sort_keys=True))
        return

    cmd = [
        sys.executable,
        "scripts/05_causal_validation.py",
        "--rollout_config",
        args.rollout_config,
        "--sae_checkpoint",
        args.sae_checkpoint,
        "--sae_config",
        args.sae_config,
        "--feature_manifest_csv",
        str(manifest),
        "--output_dir",
        str(out_dir),
        "--output_name",
        "class_mean_intervention.json",
        "--layer",
        str(int(args.layer)),
        "--num_rollouts",
        str(int(args.num_rollouts)),
        "--scale",
        "1.0",
        "--trigger_mode",
        "always",
        "--trigger_start_step",
        str(int(args.trigger_start_step)),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
