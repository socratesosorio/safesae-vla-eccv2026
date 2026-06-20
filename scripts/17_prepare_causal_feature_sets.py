"""Prepare causal-validation feature sets from monitor weight exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare top SAE features and random controls for causal runs")
    parser.add_argument("--feature_weights_csv", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_topk_frequency", type=float, default=0.0)
    parser.add_argument("--num_random_controls", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/causal_feature_sets")
    parser.add_argument("--output_prefix", type=str, default="monitor_consensus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_weights_csv)
    if "feature_idx" not in df.columns:
        raise ValueError(f"{args.feature_weights_csv} must contain feature_idx")

    if "consensus_rank" not in df.columns:
        sort_cols = [c for c in ["mean_abs_weight", "topk_frequency", "feature_idx"] if c in df.columns]
        ascending = [False if c != "feature_idx" else True for c in sort_cols]
        df = df.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
        df["consensus_rank"] = np.arange(1, len(df) + 1, dtype=np.int32)

    if "topk_frequency" in df.columns:
        df = df[df["topk_frequency"] >= float(args.min_topk_frequency)].copy()
    if df.empty:
        raise RuntimeError("No features remain after applying the requested filter")

    selected = df.head(int(args.top_k)).copy().reset_index(drop=True)
    selected["feature_set"] = "selected_topk"
    selected["rank"] = np.arange(1, len(selected) + 1, dtype=np.int32)
    selected = selected[["feature_set", "rank", "feature_idx"] + [c for c in selected.columns if c not in {"feature_set", "rank", "feature_idx"}]]
    selected.to_csv(out_dir / f"{args.output_prefix}_top_features.csv", index=False)

    universe = sorted(set(df["feature_idx"].astype(int).tolist()))
    selected_ids = set(selected["feature_idx"].astype(int).tolist())
    control_pool = [idx for idx in universe if idx not in selected_ids]
    if len(control_pool) < int(args.top_k):
        raise RuntimeError("Not enough remaining features to sample random controls")

    rng = np.random.default_rng(int(args.random_seed))
    control_rows: list[dict[str, object]] = []
    for control_idx in range(int(args.num_random_controls)):
        chosen = rng.choice(control_pool, size=int(args.top_k), replace=False)
        for rank, feat_idx in enumerate(chosen.tolist(), start=1):
            control_rows.append(
                {
                    "feature_set": f"random_control_{control_idx:02d}",
                    "rank": int(rank),
                    "feature_idx": int(feat_idx),
                }
            )

    random_df = pd.DataFrame(control_rows)
    random_df.to_csv(out_dir / f"{args.output_prefix}_random_controls.csv", index=False)

    summary = {
        "feature_weights_csv": args.feature_weights_csv,
        "top_k": int(args.top_k),
        "min_topk_frequency": float(args.min_topk_frequency),
        "num_random_controls": int(args.num_random_controls),
        "selected_feature_ids": [int(x) for x in selected["feature_idx"].tolist()],
        "random_control_feature_sets": {
            feature_set: [int(x) for x in group["feature_idx"].tolist()]
            for feature_set, group in random_df.groupby("feature_set", sort=False)
        },
    }
    (out_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
