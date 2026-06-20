"""Run all additional analyses for the ECCV paper on existing rollout data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.analysis.additional_analyses import (
    feature_inspection_table,
    latency_report,
    per_suite_breakdown,
    sparsity_performance_curve,
    temporal_violation_patterns,
)
from src.analysis.differential_activation import load_sae_checkpoint
from src.data.activation_dataset import ActivationDataset
from src.monitor.evaluate_monitor import collect_step_features
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additional ECCV analyses")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to rollout data directory")
    parser.add_argument("--sae_checkpoint", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--layer", type=int, default=20, help="Transformer layer to analyze")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results directory (for differential CSV)")
    parser.add_argument("--output_dir", type=str, default="results/additional", help="Output directory for analysis CSVs")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml", help="SAE config YAML")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(ensure_dir(args.output_dir))

    # Load SAE
    sae_cfg = load_yaml(args.sae_config)
    sae_block = sae_cfg.get("primary", sae_cfg.get("sae", sae_cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        args.sae_checkpoint,
        d_in=int(sae_block.get("d_in", 4096)),
        d_sae=int(sae_block.get("d_sae", 16384)),
        k=int(sae_block.get("k", 32)),
        device=device,
    )

    # Load dataset and extract features
    dataset = ActivationDataset(data_dir=args.data_dir, layer=args.layer, split="all")
    if len(dataset) == 0:
        raise RuntimeError(f"No rollout files found in {args.data_dir}")

    print(f"Extracting SAE features from {len(dataset)} episodes...")
    x_sae, x_raw, y_any, y_cat, forces, ep_idx, ep_success, ep_unsafe = collect_step_features(dataset, sae)
    print(f"  -> {x_sae.shape[0]} steps, {x_sae.shape[1]} SAE features")

    # Resolve feature CSV path
    results_dir = Path(args.results_dir)
    feature_csv = None
    for candidate in [
        results_dir / "analysis" / "differential" / f"layer{args.layer}_overall.csv",
        results_dir / "analysis" / "differential" / f"openvla_layer{args.layer}_overall.csv",
        results_dir / "differential" / f"layer{args.layer}_overall.csv",
    ]:
        if candidate.exists():
            feature_csv = str(candidate)
            break

    if feature_csv is None:
        raise FileNotFoundError(
            f"Could not find differential analysis CSV for layer {args.layer} in {results_dir}. "
            "Run differential_activation.py first."
        )
    print(f"Using feature ranking from: {feature_csv}")

    # Collect per-episode metadata
    metadata = []
    for i in range(len(dataset)):
        item = dataset[i]
        metadata.append(item.get("metadata", {}))

    # 1. Sparsity performance curve
    print("Running sparsity performance curve...")
    sparsity_df = sparsity_performance_curve(x_sae, y_any, y_cat, feature_csv, seed=args.seed)
    sparsity_df.to_csv(output_dir / "sparsity_curve.csv", index=False)
    print(f"  -> Saved sparsity_curve.csv ({len(sparsity_df)} rows)")

    # 2. Temporal violation patterns
    print("Running temporal violation patterns...")
    temporal_df = temporal_violation_patterns(x_sae, y_cat, ep_idx, feature_csv)
    temporal_df.to_csv(output_dir / "temporal_patterns.csv", index=False)
    print(f"  -> Saved temporal_patterns.csv ({len(temporal_df)} rows)")

    # 3. Per-suite breakdown
    print("Running per-suite breakdown...")
    suite_df = per_suite_breakdown(x_sae, y_any, y_cat, ep_idx, metadata, feature_csv, seed=args.seed)
    suite_df.to_csv(output_dir / "per_suite_breakdown.csv", index=False)
    print(f"  -> Saved per_suite_breakdown.csv ({len(suite_df)} rows)")

    # 4. Feature inspection table
    print("Running feature inspection table...")
    inspect_df = feature_inspection_table(x_sae, y_cat, ep_idx, feature_csv)
    inspect_df.to_csv(output_dir / "feature_inspection.csv", index=False)
    print(f"  -> Saved feature_inspection.csv ({len(inspect_df)} rows)")

    # 5. Latency report
    print("Running latency benchmarks...")
    latency_df = latency_report(x_sae, sae)
    latency_df.to_csv(output_dir / "latency_report.csv", index=False)
    print(f"  -> Saved latency_report.csv ({len(latency_df)} rows)")

    print(f"\nAll additional analyses saved to {output_dir}/")


if __name__ == "__main__":
    main()
