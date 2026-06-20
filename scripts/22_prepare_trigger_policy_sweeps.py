"""Prepare trigger-policy sweep manifests for a known causal feature.

The bottleneck identified from the refined-followup pass is that feature 7587
improves safety under always-on but never fires under the original trigger
policy (wrist_force_ratio >= 0.5 at start_step=10).  The calibrated goal-suite
collision_force_threshold is ~131, so the original threshold demands absolute
wrist force >= ~65.7 — far above the typical operating range.

This script sweeps over:
  - trigger thresholds (lower values to engage earlier)
  - trigger start steps (earlier engagement)
  - a small set of feature scales (gentle boosts that did not fire before)

It also adds a few always-on "anchor" rows at the same scales so we can
directly compare trigger-gated vs always-on at each scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare trigger-policy sweep manifests")
    parser.add_argument("--source_manifest_csv", type=str, required=True)
    parser.add_argument("--controls_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results/athena_benchmark_repair/causal_feature_sets")
    parser.add_argument("--output_prefix", type=str, default="trigger_policy_sweep")
    parser.add_argument("--feature_idx", type=int, default=7587)
    parser.add_argument("--target_category", type=str, default="excessive_force")
    parser.add_argument("--scales", type=str, default="1.05,1.10,1.15,1.20")
    parser.add_argument("--trigger_thresholds", type=str, default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--trigger_start_steps", type=str, default="0,2,5")
    parser.add_argument("--trigger_latch_values", type=str, default="1,0")
    parser.add_argument("--include_always_on_anchors", type=int, default=1)
    parser.add_argument("--recommended_num_rollouts", type=int, default=6)
    return parser.parse_args()


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in _parse_csv_list(value)]


def _parse_int_list(value: str) -> list[int]:
    return [int(part) for part in _parse_csv_list(value)]


def _scale_suffix(scale: float) -> str:
    text = f"{float(scale):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _build_feature_set_name(
    *,
    base: str,
    scale: float,
    trigger_mode: str,
    trigger_threshold: float | None,
    trigger_start_step: int,
    trigger_latch: bool,
) -> str:
    sc = _scale_suffix(scale)
    if trigger_mode == "always":
        return f"{base}_sc{sc}_always"
    th = _scale_suffix(trigger_threshold) if trigger_threshold is not None else "ndef"
    latch_tag = "latch" if trigger_latch else "unlatch"
    return f"{base}_sc{sc}_th{th}_s{trigger_start_step}_{latch_tag}"


def main() -> None:
    args = parse_args()
    source_df = pd.read_csv(args.source_manifest_csv)

    feature_idx = int(args.feature_idx)
    source_rows = source_df[source_df["feature_idx"].astype(int) == feature_idx].copy()
    if source_rows.empty:
        raise ValueError(f"Feature {feature_idx} not found in {args.source_manifest_csv}")

    base_row = source_rows.iloc[0].to_dict()

    scales = _parse_float_list(args.scales)
    trigger_thresholds = _parse_float_list(args.trigger_thresholds)
    trigger_start_steps = _parse_int_list(args.trigger_start_steps)
    trigger_latch_values = [bool(int(v)) for v in _parse_csv_list(args.trigger_latch_values)]
    include_anchors = bool(int(args.include_always_on_anchors))

    base_feature_set_prefix = f"{args.target_category}_protective_single_f{feature_idx}"

    rows: list[dict[str, object]] = []
    sweep_summary: list[dict[str, object]] = []

    for scale in scales:
        if include_anchors:
            name = _build_feature_set_name(
                base=base_feature_set_prefix,
                scale=scale,
                trigger_mode="always",
                trigger_threshold=None,
                trigger_start_step=0,
                trigger_latch=True,
            )
            row = dict(base_row)
            row["feature_set"] = name
            row["feature_scale"] = float(scale)
            row["trigger_mode"] = "always"
            row["trigger_threshold"] = None
            row["trigger_start_step"] = 0
            row["trigger_end_step"] = None
            row["trigger_latch"] = 1
            row["sweep_group"] = "always_on_anchor"
            row["sweep_scale"] = float(scale)
            row["sweep_trigger_mode"] = "always"
            row["sweep_trigger_threshold"] = None
            row["sweep_trigger_start_step"] = 0
            row["sweep_trigger_latch"] = True
            rows.append(row)
            sweep_summary.append({
                "feature_set": name,
                "scale": float(scale),
                "trigger_mode": "always",
                "trigger_threshold": None,
                "trigger_start_step": 0,
                "trigger_latch": True,
                "sweep_group": "always_on_anchor",
            })

        for start_step in trigger_start_steps:
            for threshold in trigger_thresholds:
                for latch in trigger_latch_values:
                    name = _build_feature_set_name(
                        base=base_feature_set_prefix,
                        scale=scale,
                        trigger_mode="wrist_force_ratio",
                        trigger_threshold=threshold,
                        trigger_start_step=start_step,
                        trigger_latch=latch,
                    )
                    row = dict(base_row)
                    row["feature_set"] = name
                    row["feature_scale"] = float(scale)
                    row["trigger_mode"] = "wrist_force_ratio"
                    row["trigger_threshold"] = float(threshold)
                    row["trigger_start_step"] = int(start_step)
                    row["trigger_end_step"] = None
                    row["trigger_latch"] = int(latch)
                    row["sweep_group"] = "trigger_gated"
                    row["sweep_scale"] = float(scale)
                    row["sweep_trigger_mode"] = "wrist_force_ratio"
                    row["sweep_trigger_threshold"] = float(threshold)
                    row["sweep_trigger_start_step"] = int(start_step)
                    row["sweep_trigger_latch"] = latch
                    rows.append(row)
                    sweep_summary.append({
                        "feature_set": name,
                        "scale": float(scale),
                        "trigger_mode": "wrist_force_ratio",
                        "trigger_threshold": float(threshold),
                        "trigger_start_step": int(start_step),
                        "trigger_latch": latch,
                        "sweep_group": "trigger_gated",
                    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.DataFrame(rows)
    sort_cols = ["sweep_group", "sweep_scale", "sweep_trigger_start_step", "sweep_trigger_threshold", "sweep_trigger_latch", "feature_set"]
    actual_sort_cols = [c for c in sort_cols if c in manifest_df.columns]
    manifest_df = manifest_df.sort_values(actual_sort_cols, kind="stable").reset_index(drop=True)
    manifest_path = out_dir / f"{args.output_prefix}_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary: dict[str, object] = {
        "source_manifest_csv": str(args.source_manifest_csv),
        "controls_csv": str(args.controls_csv).strip(),
        "feature_idx": feature_idx,
        "target_category": str(args.target_category),
        "scales": scales,
        "trigger_thresholds": trigger_thresholds,
        "trigger_start_steps": trigger_start_steps,
        "trigger_latch_values": trigger_latch_values,
        "include_always_on_anchors": include_anchors,
        "recommended_num_rollouts": int(args.recommended_num_rollouts),
        "num_feature_sets": int(manifest_df["feature_set"].nunique()),
        "num_rows": len(manifest_df),
        "manifest_csv": str(manifest_path),
        "sweep_entries": sweep_summary,
    }
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
