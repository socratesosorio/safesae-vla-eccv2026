"""Compute high/low progress episode labels from cached rollout telemetry.

Output:
  - progress_labels.csv with columns:
      episode_id,label,progress_raw,progress_norm,suite,metric_source
  - progress_label_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors import safe_open


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _extract_goal_from_metadata(meta: dict[str, Any]) -> np.ndarray | None:
    keys = [
        "goal_position",
        "target_position",
        "target_object_position",
        "goal_xyz",
        "target_xyz",
    ]
    for key in keys:
        if key in meta:
            arr = np.asarray(meta[key], dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
    return None


def _extract_progress_metric(tensor_path: Path, metadata: dict[str, Any]) -> tuple[float | None, str]:
    # 1) Direct completion metric if available (higher = better).
    for key in ("task_completion", "completion_fraction", "progress", "success_fraction"):
        val = _safe_float(metadata.get(key))
        if val is not None:
            return float(val), key

    eef_np = None
    with safe_open(str(tensor_path), framework="np") as f:
        if "eef_positions" in f.keys():
            eef_np = f.get_tensor("eef_positions").astype(np.float32).reshape(-1, 3)
    if eef_np is None:
        return None, "missing_eef_positions"

    # 2) Distance to explicit goal position if available.
    goal = _extract_goal_from_metadata(metadata)
    if goal is not None:
        dists = np.linalg.norm(eef_np - goal.reshape(1, 3), axis=1)
        progress = -float(np.min(dists))  # invert so higher = better
        return progress, "min_distance_to_goal_from_metadata"

    # 3) Optional metadata success fallback if provided.
    for key in ("episode_success", "success"):
        val = metadata.get(key, None)
        if isinstance(val, bool):
            return (1.0 if val else 0.0), key
        fval = _safe_float(val)
        if fval is not None:
            return float(fval), key

    # 4) Last-resort geometric fallback: negative final EE displacement magnitude.
    # Not semantically ideal but gives deterministic ordering when telemetry is limited.
    if eef_np.shape[0] >= 2:
        disp = np.linalg.norm(eef_np[-1] - eef_np[0])
        return -float(disp), "negative_eef_displacement_fallback"
    return None, "no_metric_available"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute quartile-based progress labels")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with rollout_*.safetensors + rollout_*.json")
    p.add_argument("--output_dir", type=str, default="results/progress_labels")
    p.add_argument("--top_frac", type=float, default=0.25, help="Top fraction for high_progress label")
    p.add_argument("--bottom_frac", type=float, default=0.25, help="Bottom fraction for low_progress label")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor_files = sorted(data_dir.rglob("rollout_*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No rollout_*.safetensors found in {data_dir}")

    rows: list[dict[str, Any]] = []
    for tensor_path in tensor_files:
        episode_id = tensor_path.stem
        meta_path = tensor_path.with_suffix(".json")
        metadata = _read_json(meta_path) if meta_path.exists() else {}
        suite = str(metadata.get("suite", "unknown"))
        metric, source = _extract_progress_metric(tensor_path, metadata)
        if metric is None:
            continue
        rows.append(
            {
                "episode_id": episode_id,
                "suite": suite,
                "progress_raw": float(metric),
                "metric_source": source,
            }
        )

    if not rows:
        raise RuntimeError("Could not derive any progress metric from provided episodes.")

    df = pd.DataFrame(rows)

    # Normalize within suite to reduce scale mismatch across suites.
    suite_norm = []
    for suite, gdf in df.groupby("suite"):
        vals = gdf["progress_raw"].to_numpy()
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin + 1e-12:
            norm = np.full_like(vals, 0.5, dtype=np.float64)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        gdf = gdf.copy()
        gdf["progress_norm"] = norm
        suite_norm.append(gdf)
    df = pd.concat(suite_norm, axis=0).sort_values("episode_id").reset_index(drop=True)

    # Global quartile split on normalized progress.
    n = len(df)
    n_low = int(np.floor(n * float(args.bottom_frac)))
    n_high = int(np.floor(n * float(args.top_frac)))
    if n_low < 1 or n_high < 1:
        raise RuntimeError(f"Dataset too small for requested split: n={n}, low={n_low}, high={n_high}")

    sorted_idx = np.argsort(df["progress_norm"].to_numpy())
    low_ids = set(df.iloc[sorted_idx[:n_low]]["episode_id"].tolist())
    high_ids = set(df.iloc[sorted_idx[-n_high:]]["episode_id"].tolist())

    labels = []
    for ep_id in df["episode_id"].tolist():
        if ep_id in high_ids:
            labels.append(1)
        elif ep_id in low_ids:
            labels.append(0)
        else:
            labels.append(-1)  # discarded middle split
    df["label"] = labels

    # Persist full table and compact label CSV for analysis scripts.
    full_csv = out_dir / "progress_labels_full.csv"
    label_csv = out_dir / "progress_labels.csv"
    df.to_csv(full_csv, index=False)
    df[df["label"].isin([0, 1])][["episode_id", "label"]].to_csv(label_csv, index=False)

    summary = {
        "num_episodes_total": int(len(df)),
        "num_labeled_low": int((df["label"] == 0).sum()),
        "num_labeled_high": int((df["label"] == 1).sum()),
        "num_discarded_middle": int((df["label"] == -1).sum()),
        "suites": sorted(df["suite"].unique().tolist()),
        "metric_source_counts": df["metric_source"].value_counts().to_dict(),
        "output_csv": str(label_csv),
        "output_full_csv": str(full_csv),
    }
    with (out_dir / "progress_label_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
