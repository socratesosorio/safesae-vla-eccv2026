"""Prepare gentler causal follow-up manifests from prior causal control results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare refined causal follow-up manifests")
    parser.add_argument("--controls_csv", type=str, required=True)
    parser.add_argument("--source_manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/causal_feature_sets")
    parser.add_argument("--output_prefix", type=str, default="refined_causal_followups")
    parser.add_argument("--target_category", type=str, default="")
    parser.add_argument("--feature_sets", type=str, default="")
    parser.add_argument("--max_candidates", type=int, default=1)
    parser.add_argument("--max_success_drop", type=float, default=0.25)
    parser.add_argument("--min_target_improvement", type=float, default=0.05)
    parser.add_argument("--min_any_violation_improvement", type=float, default=0.0)
    parser.add_argument("--require_action_delta", type=int, default=1)
    parser.add_argument("--gentle_boost_scales", type=str, default="1.02,1.05,1.1,1.15,1.2")
    parser.add_argument("--gentle_suppress_scales", type=str, default="0.75,0.5,0.25")
    parser.add_argument("--inherit_source_trigger", type=int, default=1)
    parser.add_argument("--recommended_num_rollouts", type=int, default=6)
    return parser.parse_args()


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in _parse_csv_list(value)]


def _scale_suffix(scale: float) -> str:
    text = f"{float(scale):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _base_feature_set_name(feature_set: str) -> str:
    text = str(feature_set).strip()
    if text.endswith("_always_on"):
        return text[: -len("_always_on")]
    return text


def _candidate_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["success_drop"] = out["success_rate_baseline"].astype(float) - out["success_rate_clamped"].astype(float)
    out["target_improvement"] = (
        out["target_category_rate_baseline"].astype(float) - out["target_category_rate_clamped"].astype(float)
    )
    out["any_violation_improvement"] = (
        out["any_violation_rate_baseline"].astype(float) - out["any_violation_rate_clamped"].astype(float)
    )
    out["action_delta_rate"] = out.get("action_delta_any_nonzero_rate_clamped", 0.0).astype(float)
    out["base_feature_set"] = out["feature_set"].astype(str).map(_base_feature_set_name)
    return out


def _filter_scales(*, tested_scale: float, direction: str, candidate_scales: list[float]) -> list[float]:
    neutral = 1.0
    direction = str(direction).strip().lower()
    if direction == "boost":
        scales = [float(scale) for scale in candidate_scales if neutral < float(scale) < float(tested_scale)]
    else:
        scales = [float(scale) for scale in candidate_scales if float(tested_scale) < float(scale) < neutral]
    return list(dict.fromkeys(scales))


def main() -> None:
    args = parse_args()
    controls_df = pd.read_csv(args.controls_csv)
    source_manifest_df = pd.read_csv(args.source_manifest_csv)
    required_controls = {
        "feature_set",
        "success_rate_clamped",
        "success_rate_baseline",
        "any_violation_rate_clamped",
        "any_violation_rate_baseline",
        "target_category_rate_clamped",
        "target_category_rate_baseline",
        "intervention_direction",
        "feature_scale_mean",
    }
    missing_controls = required_controls - set(controls_df.columns)
    if missing_controls:
        raise ValueError(f"{args.controls_csv} missing required columns: {sorted(missing_controls)}")
    required_source = {"feature_set", "feature_idx", "feature_scale"}
    missing_source = required_source - set(source_manifest_df.columns)
    if missing_source:
        raise ValueError(f"{args.source_manifest_csv} missing required columns: {sorted(missing_source)}")

    controls_df = _candidate_scores(controls_df)
    if str(args.target_category).strip():
        controls_df = controls_df[controls_df["target_category"].astype(str) == str(args.target_category).strip()].copy()
    requested_feature_sets = set(_parse_csv_list(args.feature_sets))
    if requested_feature_sets:
        requested_bases = {_base_feature_set_name(name) for name in requested_feature_sets}
        controls_df = controls_df[controls_df["base_feature_set"].isin(requested_bases)].copy()
    controls_df = controls_df[controls_df.get("is_random_control", 0).astype(int) == 0].copy()
    if bool(int(args.require_action_delta)):
        controls_df = controls_df[controls_df["action_delta_rate"] > 0.0].copy()
    if controls_df.empty:
        raise RuntimeError("No causal-control rows remain after applying the requested follow-up filters")

    strict_df = controls_df[
        (controls_df["target_improvement"] >= float(args.min_target_improvement))
        & (controls_df["any_violation_improvement"] >= float(args.min_any_violation_improvement))
        & (controls_df["success_drop"] <= float(args.max_success_drop))
    ].copy()
    strict_df["selection_mode"] = "strict_success_preserving"

    fallback_df = controls_df[
        (controls_df["target_improvement"] >= float(args.min_target_improvement))
        & (controls_df["any_violation_improvement"] >= float(args.min_any_violation_improvement))
    ].copy()
    fallback_df["selection_mode"] = "fallback_positive_but_too_harmful"

    selected_df = strict_df if not strict_df.empty else fallback_df
    if selected_df.empty:
        raise RuntimeError("No follow-up candidates improved the target category under the requested thresholds")
    selected_df = selected_df.sort_values(
        ["target_improvement", "any_violation_improvement", "success_drop", "action_delta_rate", "feature_set"],
        ascending=[False, False, True, False, True],
        kind="stable",
    ).head(int(args.max_candidates))

    boost_scales = _parse_float_list(args.gentle_boost_scales)
    suppress_scales = _parse_float_list(args.gentle_suppress_scales)

    rows: list[dict[str, object]] = []
    selected_summary_rows: list[dict[str, object]] = []
    for candidate in selected_df.itertuples(index=False):
        base_feature_set = str(candidate.base_feature_set)
        source_rows = source_manifest_df[source_manifest_df["feature_set"].astype(str) == base_feature_set].copy()
        if source_rows.empty:
            raise RuntimeError(f"Base feature set {base_feature_set!r} not found in {args.source_manifest_csv}")
        direction = str(candidate.intervention_direction).strip().lower()
        tested_scale = float(candidate.feature_scale_mean)
        proposed_scales = _filter_scales(
            tested_scale=tested_scale,
            direction=direction,
            candidate_scales=boost_scales if direction == "boost" else suppress_scales,
        )
        if not proposed_scales:
            raise RuntimeError(
                f"No gentler scales remain for {base_feature_set!r} under direction={direction!r} and tested_scale={tested_scale}"
            )
        for scale in proposed_scales:
            feature_set_name = f"{base_feature_set}_refine_sc{_scale_suffix(scale)}"
            for row in source_rows.itertuples(index=False):
                out_row = {column: getattr(row, column) for column in source_rows.columns}
                out_row["feature_set"] = feature_set_name
                out_row["feature_scale"] = float(scale)
                out_row["selection_strategy"] = "refined_followup"
                out_row["refine_parent_feature_set"] = str(candidate.feature_set)
                out_row["refine_base_feature_set"] = base_feature_set
                out_row["refine_parent_scale"] = float(tested_scale)
                out_row["refine_selection_mode"] = str(candidate.selection_mode)
                out_row["refine_success_drop"] = float(candidate.success_drop)
                out_row["refine_target_improvement"] = float(candidate.target_improvement)
                out_row["refine_any_violation_improvement"] = float(candidate.any_violation_improvement)
                out_row["refine_action_delta_rate"] = float(candidate.action_delta_rate)
                if not bool(int(args.inherit_source_trigger)):
                    out_row["trigger_mode"] = "always"
                    out_row["trigger_threshold"] = None
                    out_row["trigger_start_step"] = 0
                    out_row["trigger_end_step"] = None
                    out_row["trigger_latch"] = 1
                rows.append(out_row)
        selected_summary_rows.append(
            {
                "feature_set": str(candidate.feature_set),
                "base_feature_set": base_feature_set,
                "selection_mode": str(candidate.selection_mode),
                "success_drop": float(candidate.success_drop),
                "target_improvement": float(candidate.target_improvement),
                "any_violation_improvement": float(candidate.any_violation_improvement),
                "action_delta_rate": float(candidate.action_delta_rate),
                "tested_scale": float(tested_scale),
                "proposed_scales": proposed_scales,
                "intervention_direction": str(candidate.intervention_direction),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(rows).sort_values(
        ["target_category", "feature_set", "rank"],
        ascending=[True, True, True],
        kind="stable",
    )
    manifest_path = out_dir / f"{args.output_prefix}_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "controls_csv": str(args.controls_csv),
        "source_manifest_csv": str(args.source_manifest_csv),
        "manifest_csv": str(manifest_path),
        "target_category_filter": str(args.target_category).strip(),
        "requested_feature_sets": sorted(requested_feature_sets),
        "max_candidates": int(args.max_candidates),
        "max_success_drop": float(args.max_success_drop),
        "min_target_improvement": float(args.min_target_improvement),
        "min_any_violation_improvement": float(args.min_any_violation_improvement),
        "require_action_delta": bool(int(args.require_action_delta)),
        "inherit_source_trigger": bool(int(args.inherit_source_trigger)),
        "recommended_num_rollouts": int(args.recommended_num_rollouts),
        "selected_candidates": selected_summary_rows,
        "num_feature_sets": int(manifest_df["feature_set"].nunique()),
    }
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
