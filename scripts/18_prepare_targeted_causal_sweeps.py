"""Build sign-aware, category-targeted causal intervention manifests."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SAFETY_CATEGORIES = [
    "collision",
    "excessive_force",
    "boundary_violation",
    "high_approach_speed",
    "object_drop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare targeted causal sweep manifests")
    parser.add_argument("--feature_weights_csv", type=str, required=True)
    parser.add_argument("--category_feature_dir", type=str, required=True)
    parser.add_argument("--categories", type=str, default=",".join(SAFETY_CATEGORIES))
    parser.add_argument("--min_topk_frequency", type=float, default=0.25)
    parser.add_argument("--min_sign_alignment", type=float, default=0.60)
    parser.add_argument("--max_risky_single_features", type=int, default=4)
    parser.add_argument("--max_protective_single_features", type=int, default=2)
    parser.add_argument("--risky_pair_pool_size", type=int, default=3)
    parser.add_argument("--protective_pair_pool_size", type=int, default=2)
    parser.add_argument("--risky_single_scale", type=float, default=0.25)
    parser.add_argument("--protective_single_scale", type=float, default=1.25)
    parser.add_argument("--risky_pair_scale", type=float, default=0.50)
    parser.add_argument("--protective_pair_scale", type=float, default=1.15)
    parser.add_argument("--suppress_scales", type=str, default="0.0,0.25,0.5,0.75")
    parser.add_argument("--boost_scales", type=str, default="1.1,1.25,1.5")
    parser.add_argument("--include_risky_bundles", action="store_true")
    parser.add_argument("--risky_bundle_size", type=int, default=3)
    parser.add_argument("--risky_bundle_scale", type=float, default=0.50)
    parser.add_argument("--allowed_suites", type=str, default="")
    parser.add_argument("--allowed_task_specs", type=str, default="")
    parser.add_argument("--screened_cells_csv", type=str, default="")
    parser.add_argument("--screened_top_k_cells_per_category", type=int, default=4)
    parser.add_argument("--trigger_start_step", type=int, default=10)
    parser.add_argument("--trigger_end_step", type=int, default=-1)
    parser.add_argument("--trigger_latch", type=int, default=1)
    parser.add_argument("--boundary_trigger_margin", type=float, default=0.05)
    parser.add_argument("--force_trigger_ratio", type=float, default=0.50)
    parser.add_argument("--speed_trigger_ratio", type=float, default=0.80)
    parser.add_argument("--output_dir", type=str, default="results/causal_feature_sets")
    parser.add_argument("--output_prefix", type=str, default="targeted_causal_sweeps")
    return parser.parse_args()


def _normalize_series(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 1e-12:
        return pd.Series(np.ones(len(arr), dtype=np.float64), index=arr.index)
    return (arr - lo) / (hi - lo)


def _load_category_df(category_feature_dir: Path, category: str) -> pd.DataFrame:
    candidates = sorted(category_feature_dir.glob(f"*_{category}.csv"))
    if not candidates:
        raise FileNotFoundError(f"No category feature CSV found for {category} under {category_feature_dir}")
    return pd.read_csv(candidates[0])


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_scales(value: str, fallback: list[float]) -> list[float]:
    parts = _parse_csv_list(value)
    if not parts:
        return [float(x) for x in fallback]
    return [float(part) for part in parts]


def _scale_suffix(scale: float) -> str:
    text = f"{float(scale):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _load_screened_cells(screened_cells_csv: str) -> pd.DataFrame:
    path = str(screened_cells_csv).strip()
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"target_category", "suite", "task_idx", "condition", "recommended"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df


def _allowlists_for_category(
    *,
    screened_df: pd.DataFrame,
    category: str,
    top_k: int,
    default_allowed_suites: str,
    default_allowed_task_specs: str,
) -> tuple[str, str, list[str]]:
    if screened_df.empty:
        return default_allowed_suites, default_allowed_task_specs, []
    subset = screened_df[
        (screened_df["target_category"].astype(str) == str(category))
        & (screened_df["recommended"].astype(bool))
    ].copy()
    if subset.empty:
        return default_allowed_suites, default_allowed_task_specs, []
    subset = subset.sort_values(
        ["success_rate", "clean_success_rate", "target_category_rate", "num_episodes", "suite", "task_idx"],
        ascending=[False, False, True, False, True, True],
        kind="stable",
    ).head(int(top_k))
    allowed_suites = sorted({str(x) for x in subset["suite"].tolist()})
    allowed_task_specs = [f"{str(row.suite)}:{int(row.task_idx)}" for row in subset.itertuples(index=False)]
    condition_names = sorted({str(x) for x in subset["condition"].tolist()})
    if default_allowed_suites:
        allowed_suites = sorted(set(allowed_suites).union(_parse_csv_list(default_allowed_suites)))
    if default_allowed_task_specs:
        allowed_task_specs = sorted(set(allowed_task_specs).union(_parse_csv_list(default_allowed_task_specs)))
    return ",".join(allowed_suites), ",".join(allowed_task_specs), condition_names


def _trigger_defaults_for_category(
    category: str,
    *,
    trigger_start_step: int,
    trigger_end_step: int,
    trigger_latch: int,
    boundary_trigger_margin: float,
    force_trigger_ratio: float,
    speed_trigger_ratio: float,
) -> dict[str, object]:
    trigger_end = None if int(trigger_end_step) < 0 else int(trigger_end_step)
    base = {
        "trigger_start_step": int(trigger_start_step),
        "trigger_end_step": trigger_end,
        "trigger_latch": int(bool(trigger_latch)),
    }
    if category == "boundary_violation":
        return {
            **base,
            "trigger_mode": "boundary_margin",
            "trigger_threshold": float(boundary_trigger_margin),
        }
    if category in {"collision", "excessive_force"}:
        return {
            **base,
            "trigger_mode": "wrist_force_ratio",
            "trigger_threshold": float(force_trigger_ratio),
        }
    if category == "high_approach_speed":
        return {
            **base,
            "trigger_mode": "eef_speed_ratio",
            "trigger_threshold": float(speed_trigger_ratio),
        }
    return {
        **base,
        "trigger_mode": "always",
        "trigger_threshold": None,
    }


def _build_ranked_candidates(
    *,
    monitor_df: pd.DataFrame,
    category_df: pd.DataFrame,
    category: str,
    min_sign_alignment: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = monitor_df.merge(category_df, on="feature_idx", how="inner", suffixes=("_monitor", "_category"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["category_score_norm"] = _normalize_series(merged["composite_score"].fillna(0.0))
    merged["monitor_weight_norm"] = _normalize_series(
        merged["mean_normalized_abs_weight"] if "mean_normalized_abs_weight" in merged.columns else merged["mean_abs_weight"]
    )
    merged["topk_frequency_norm"] = _normalize_series(merged["topk_frequency"].fillna(0.0))
    merged["rank_score_norm"] = 1.0 / merged["consensus_rank"].astype(float).clip(lower=1.0)
    merged["risky_sign_alignment"] = merged["positive_weight_fraction"].astype(float)
    merged["protective_sign_alignment"] = 1.0 - merged["positive_weight_fraction"].astype(float)

    risky = merged[
        (merged["direction"] == "higher_in_unsafe")
        & (merged["mean_signed_weight"] > 0.0)
        & (merged["risky_sign_alignment"] >= float(min_sign_alignment))
    ].copy()
    if not risky.empty:
        risky["selection_score"] = (
            0.45 * risky["category_score_norm"]
            + 0.25 * risky["topk_frequency_norm"]
            + 0.20 * risky["risky_sign_alignment"]
            + 0.10 * risky["rank_score_norm"]
        )
        risky["target_category"] = category
        risky = risky.sort_values(
            ["selection_score", "composite_score", "consensus_rank", "feature_idx"],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)

    protective = merged[
        (merged["direction"] == "higher_in_safe")
        & (merged["mean_signed_weight"] < 0.0)
        & (merged["protective_sign_alignment"] >= float(min_sign_alignment))
    ].copy()
    if not protective.empty:
        protective["selection_score"] = (
            0.45 * protective["category_score_norm"]
            + 0.25 * protective["topk_frequency_norm"]
            + 0.20 * protective["protective_sign_alignment"]
            + 0.10 * protective["rank_score_norm"]
        )
        protective["target_category"] = category
        protective = protective.sort_values(
            ["selection_score", "composite_score", "consensus_rank", "feature_idx"],
            ascending=[False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)

    return risky, protective


def _append_feature_set_rows(
    *,
    rows: list[dict[str, object]],
    feature_set: str,
    feature_ids: list[int],
    feature_scale: float,
    target_category: str,
    intervention_direction: str,
    selection_strategy: str,
    allowed_suites: str,
    allowed_task_specs: str,
    condition_names: list[str],
    trigger_defaults: dict[str, object],
) -> None:
    for rank, feature_idx in enumerate(feature_ids, start=1):
        rows.append(
            {
                "feature_set": feature_set,
                "rank": int(rank),
                "feature_idx": int(feature_idx),
                "feature_scale": float(feature_scale),
                "target_category": target_category,
                "hazard_category": target_category,
                "condition_group": "hazard_targeted",
                "condition_names": ",".join(condition_names or [f"hazard_{target_category}"]),
                "intervention_direction": intervention_direction,
                "selection_strategy": selection_strategy,
                "allowed_suites": allowed_suites,
                "allowed_task_specs": allowed_task_specs,
                "trigger_mode": trigger_defaults.get("trigger_mode", "always"),
                "trigger_threshold": trigger_defaults.get("trigger_threshold", None),
                "trigger_start_step": int(trigger_defaults.get("trigger_start_step", 0)),
                "trigger_end_step": trigger_defaults.get("trigger_end_step", None),
                "trigger_latch": int(trigger_defaults.get("trigger_latch", 1)),
                "is_random_control": 0,
            }
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_df = pd.read_csv(args.feature_weights_csv)
    required_cols = {"feature_idx", "consensus_rank", "topk_frequency", "positive_weight_fraction", "mean_signed_weight"}
    missing = required_cols - set(monitor_df.columns)
    if missing:
        raise ValueError(f"{args.feature_weights_csv} missing required columns: {sorted(missing)}")

    monitor_df = monitor_df[monitor_df["topk_frequency"] >= float(args.min_topk_frequency)].copy()
    if monitor_df.empty:
        raise RuntimeError("No monitor features remain after applying min_topk_frequency")

    category_feature_dir = Path(args.category_feature_dir)
    categories = [cat.strip() for cat in args.categories.split(",") if cat.strip()]
    allowed_suites = str(args.allowed_suites).strip()
    allowed_task_specs = str(args.allowed_task_specs).strip()
    screened_df = _load_screened_cells(args.screened_cells_csv)
    suppress_scales = _parse_float_scales(
        args.suppress_scales,
        fallback=[float(args.risky_single_scale), float(args.risky_pair_scale)],
    )
    boost_scales = _parse_float_scales(
        args.boost_scales,
        fallback=[float(args.protective_single_scale), float(args.protective_pair_scale)],
    )

    manifest_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "feature_weights_csv": args.feature_weights_csv,
        "category_feature_dir": str(category_feature_dir),
        "categories": categories,
        "min_topk_frequency": float(args.min_topk_frequency),
        "min_sign_alignment": float(args.min_sign_alignment),
        "allowed_suites": allowed_suites,
        "allowed_task_specs": allowed_task_specs,
        "screened_cells_csv": str(args.screened_cells_csv).strip(),
        "suppress_scales": suppress_scales,
        "boost_scales": boost_scales,
        "per_category": {},
    }

    for category in categories:
        category_allowed_suites, category_allowed_task_specs, category_condition_names = _allowlists_for_category(
            screened_df=screened_df,
            category=category,
            top_k=int(args.screened_top_k_cells_per_category),
            default_allowed_suites=allowed_suites,
            default_allowed_task_specs=allowed_task_specs,
        )
        trigger_defaults = _trigger_defaults_for_category(
            category,
            trigger_start_step=int(args.trigger_start_step),
            trigger_end_step=int(args.trigger_end_step),
            trigger_latch=int(args.trigger_latch),
            boundary_trigger_margin=float(args.boundary_trigger_margin),
            force_trigger_ratio=float(args.force_trigger_ratio),
            speed_trigger_ratio=float(args.speed_trigger_ratio),
        )
        category_df = _load_category_df(category_feature_dir, category)
        risky, protective = _build_ranked_candidates(
            monitor_df=monitor_df,
            category_df=category_df,
            category=category,
            min_sign_alignment=float(args.min_sign_alignment),
        )

        risky_top = risky.head(int(args.max_risky_single_features)).copy()
        protective_top = protective.head(int(args.max_protective_single_features)).copy()

        for row in risky_top.itertuples(index=False):
            feat_idx = int(row.feature_idx)
            for scale in suppress_scales:
                _append_feature_set_rows(
                    rows=manifest_rows,
                    feature_set=f"{category}_risky_single_f{feat_idx}_sc{_scale_suffix(scale)}",
                    feature_ids=[feat_idx],
                    feature_scale=float(scale),
                    target_category=category,
                    intervention_direction="suppress",
                    selection_strategy="category_risky_single",
                    allowed_suites=category_allowed_suites,
                    allowed_task_specs=category_allowed_task_specs,
                    condition_names=category_condition_names,
                    trigger_defaults=trigger_defaults,
                )

        for row in protective_top.itertuples(index=False):
            feat_idx = int(row.feature_idx)
            for scale in boost_scales:
                _append_feature_set_rows(
                    rows=manifest_rows,
                    feature_set=f"{category}_protective_single_f{feat_idx}_sc{_scale_suffix(scale)}",
                    feature_ids=[feat_idx],
                    feature_scale=float(scale),
                    target_category=category,
                    intervention_direction="boost",
                    selection_strategy="category_protective_single",
                    allowed_suites=category_allowed_suites,
                    allowed_task_specs=category_allowed_task_specs,
                    condition_names=category_condition_names,
                    trigger_defaults=trigger_defaults,
                )

        risky_pair_pool = risky.head(int(args.risky_pair_pool_size))["feature_idx"].astype(int).tolist()
        for left, right in combinations(risky_pair_pool, 2):
            for scale in suppress_scales:
                _append_feature_set_rows(
                    rows=manifest_rows,
                    feature_set=f"{category}_risky_pair_f{left}_f{right}_sc{_scale_suffix(scale)}",
                    feature_ids=[int(left), int(right)],
                    feature_scale=float(scale),
                    target_category=category,
                    intervention_direction="suppress",
                    selection_strategy="category_risky_pair",
                    allowed_suites=category_allowed_suites,
                    allowed_task_specs=category_allowed_task_specs,
                    condition_names=category_condition_names,
                    trigger_defaults=trigger_defaults,
                )

        protective_pair_pool = protective.head(int(args.protective_pair_pool_size))["feature_idx"].astype(int).tolist()
        for left, right in combinations(protective_pair_pool, 2):
            for scale in boost_scales:
                _append_feature_set_rows(
                    rows=manifest_rows,
                    feature_set=f"{category}_protective_pair_f{left}_f{right}_sc{_scale_suffix(scale)}",
                    feature_ids=[int(left), int(right)],
                    feature_scale=float(scale),
                    target_category=category,
                    intervention_direction="boost",
                    selection_strategy="category_protective_pair",
                    allowed_suites=category_allowed_suites,
                    allowed_task_specs=category_allowed_task_specs,
                    condition_names=category_condition_names,
                    trigger_defaults=trigger_defaults,
                )

        if args.include_risky_bundles:
            risky_bundle = risky.head(int(args.risky_bundle_size))["feature_idx"].astype(int).tolist()
            if len(risky_bundle) >= 2:
                _append_feature_set_rows(
                    rows=manifest_rows,
                    feature_set=f"{category}_risky_bundle_top{len(risky_bundle)}_sc{_scale_suffix(float(args.risky_bundle_scale))}",
                    feature_ids=risky_bundle,
                    feature_scale=float(args.risky_bundle_scale),
                    target_category=category,
                    intervention_direction="suppress",
                    selection_strategy="category_risky_bundle",
                    allowed_suites=category_allowed_suites,
                    allowed_task_specs=category_allowed_task_specs,
                    condition_names=category_condition_names,
                    trigger_defaults=trigger_defaults,
                )

        summary["per_category"][category] = {
            "num_risky_candidates": int(len(risky)),
            "num_protective_candidates": int(len(protective)),
            "selected_risky_single_features": risky_top["feature_idx"].astype(int).tolist(),
            "selected_protective_single_features": protective_top["feature_idx"].astype(int).tolist(),
            "risky_pair_pool": risky_pair_pool,
            "protective_pair_pool": protective_pair_pool,
            "allowed_suites": _parse_csv_list(category_allowed_suites),
            "allowed_task_specs": _parse_csv_list(category_allowed_task_specs),
            "condition_names": category_condition_names,
            "trigger_defaults": trigger_defaults,
        }

    if not manifest_rows:
        raise RuntimeError("No targeted causal feature sets were generated")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ["target_category", "selection_strategy", "feature_set", "rank"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    manifest_path = out_dir / f"{args.output_prefix}_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary["num_feature_sets"] = int(manifest_df["feature_set"].nunique())
    summary["manifest_csv"] = str(manifest_path)
    (out_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
