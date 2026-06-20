"""Prepare tiny always-on causal manifests for intervention sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare always-on sanity manifests from an existing causal manifest")
    parser.add_argument("--feature_manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/causal_feature_sets")
    parser.add_argument("--output_prefix", type=str, default="always_on_sanity")
    parser.add_argument("--feature_sets", type=str, default="")
    parser.add_argument("--target_category", type=str, default="")
    parser.add_argument("--intervention_direction", type=str, default="")
    parser.add_argument("--selection_strategy", type=str, default="")
    parser.add_argument("--max_feature_sets", type=int, default=2)
    parser.add_argument("--prefer_single_features", type=int, default=1)
    parser.add_argument("--trigger_start_step", type=int, default=0)
    parser.add_argument("--trigger_end_step", type=int, default=-1)
    parser.add_argument("--trigger_latch", type=int, default=1)
    parser.add_argument("--recommended_num_rollouts", type=int, default=4)
    return parser.parse_args()


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _feature_set_order(manifest_df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(manifest_df["feature_set"].astype(str).tolist()))


def _subset_feature_sets(
    manifest_df: pd.DataFrame,
    *,
    feature_sets: list[str],
    target_category: str,
    intervention_direction: str,
    selection_strategy: str,
    max_feature_sets: int,
    prefer_single_features: bool,
) -> list[str]:
    grouped = manifest_df.groupby("feature_set", sort=False)
    stats_rows: list[dict[str, object]] = []
    for feature_set, group in grouped:
        stats_rows.append(
            {
                "feature_set": str(feature_set),
                "target_category": str(group["target_category"].iloc[0]) if "target_category" in group.columns else "",
                "intervention_direction": str(group["intervention_direction"].iloc[0]) if "intervention_direction" in group.columns else "",
                "selection_strategy": str(group["selection_strategy"].iloc[0]) if "selection_strategy" in group.columns else "",
                "num_features": int(group["feature_idx"].nunique()),
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    if feature_sets:
        requested = set(feature_sets)
        stats_df = stats_df[stats_df["feature_set"].isin(requested)].copy()
    if target_category:
        stats_df = stats_df[stats_df["target_category"].astype(str) == str(target_category)].copy()
    if intervention_direction:
        stats_df = stats_df[stats_df["intervention_direction"].astype(str) == str(intervention_direction)].copy()
    if selection_strategy:
        stats_df = stats_df[stats_df["selection_strategy"].astype(str) == str(selection_strategy)].copy()
    if prefer_single_features:
        singles = stats_df[stats_df["num_features"] == 1].copy()
        if not singles.empty:
            stats_df = singles
    feature_order = _feature_set_order(manifest_df)
    rank_map = {name: idx for idx, name in enumerate(feature_order)}
    stats_df["manifest_rank"] = stats_df["feature_set"].map(rank_map).fillna(len(rank_map)).astype(int)
    stats_df = stats_df.sort_values(["manifest_rank", "feature_set"], kind="stable")
    if max_feature_sets > 0:
        stats_df = stats_df.head(int(max_feature_sets)).copy()
    return stats_df["feature_set"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    manifest_df = pd.read_csv(args.feature_manifest_csv)
    required = {"feature_set", "feature_idx"}
    missing = required - set(manifest_df.columns)
    if missing:
        raise ValueError(f"{args.feature_manifest_csv} missing required columns: {sorted(missing)}")

    selected_feature_sets = _subset_feature_sets(
        manifest_df,
        feature_sets=_parse_csv_list(args.feature_sets),
        target_category=str(args.target_category).strip(),
        intervention_direction=str(args.intervention_direction).strip(),
        selection_strategy=str(args.selection_strategy).strip(),
        max_feature_sets=int(args.max_feature_sets),
        prefer_single_features=bool(int(args.prefer_single_features)),
    )
    if not selected_feature_sets:
        raise RuntimeError("No feature sets matched the requested always-on sanity filters")

    subset = manifest_df[manifest_df["feature_set"].astype(str).isin(selected_feature_sets)].copy()
    trigger_end = None if int(args.trigger_end_step) < 0 else int(args.trigger_end_step)
    subset["feature_set"] = subset["feature_set"].astype(str) + "_always_on"
    subset["trigger_mode"] = "always"
    subset["trigger_threshold"] = None
    subset["trigger_start_step"] = int(args.trigger_start_step)
    subset["trigger_end_step"] = trigger_end
    subset["trigger_latch"] = int(bool(args.trigger_latch))
    subset["sanity_mode"] = "always_on"
    subset["sanity_source_feature_set"] = (
        subset["feature_set"].astype(str).str.replace("_always_on", "", regex=False)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{args.output_prefix}_manifest.csv"
    subset.to_csv(manifest_path, index=False)

    summary = {
        "source_manifest_csv": str(args.feature_manifest_csv),
        "manifest_csv": str(manifest_path),
        "selected_feature_sets": selected_feature_sets,
        "output_feature_sets": sorted(subset["feature_set"].astype(str).unique().tolist()),
        "target_category_filter": str(args.target_category).strip(),
        "intervention_direction_filter": str(args.intervention_direction).strip(),
        "selection_strategy_filter": str(args.selection_strategy).strip(),
        "prefer_single_features": bool(int(args.prefer_single_features)),
        "max_feature_sets": int(args.max_feature_sets),
        "trigger_mode": "always",
        "trigger_threshold": None,
        "trigger_start_step": int(args.trigger_start_step),
        "trigger_end_step": trigger_end,
        "trigger_latch": bool(int(args.trigger_latch)),
        "recommended_num_rollouts": int(args.recommended_num_rollouts),
    }
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
