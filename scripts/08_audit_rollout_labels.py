"""Audit rollout safety labels and refresh onset metadata for cached rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_utils import (  # noqa: E402
    DEFAULT_EVENT_MIN_ACTIVE_STEPS,
    build_temporal_event_metadata,
)
from src.data.safety_labeler import SAFETY_CATEGORIES  # noqa: E402
from src.data.safety_labeler import SafetyLabeler  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--rewrite_metadata",
        action="store_true",
        help="Write refreshed onset metadata back into rollout_*.json sidecars.",
    )
    parser.add_argument(
        "--relabel_from_tensors",
        action="store_true",
        help="Recompute safety labels from actions/eef_positions/contact_forces using SafetyLabeler.",
    )
    parser.add_argument(
        "--event_min_steps",
        nargs="*",
        default=[],
        help="Optional per-category onset dwell overrides like collision=4 boundary_violation=8",
    )
    parser.add_argument("--collision_force_threshold", type=float, default=None)
    parser.add_argument("--excessive_force_threshold", type=float, default=None)
    parser.add_argument("--speed_threshold", type=float, default=None)
    parser.add_argument("--drop_velocity_threshold", type=float, default=None)
    parser.add_argument("--boundary_x", nargs=2, type=float, default=None)
    parser.add_argument("--boundary_y", nargs=2, type=float, default=None)
    parser.add_argument("--boundary_z", nargs=2, type=float, default=None)
    return parser.parse_args()


def parse_event_min_steps(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --event_min_steps entry: {item}")
        key, value = item.split("=", 1)
        overrides[str(key)] = max(int(value), 1)
    return overrides


def normalize_safety_labels(tensors: dict[str, torch.Tensor], num_categories: int) -> np.ndarray:
    labels = tensors.get("safety_labels", torch.zeros((0, num_categories), dtype=torch.bool)).to(torch.bool)
    if labels.ndim == 3:
        labels = labels.any(dim=1)
    labels_np = labels.cpu().numpy()
    if labels_np.ndim != 2:
        return np.zeros((0, num_categories), dtype=bool)
    return labels_np.astype(bool)


def relabel_from_tensors(tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> np.ndarray:
    safety_cfg = {}
    if args.collision_force_threshold is not None:
        safety_cfg["collision_force_threshold"] = float(args.collision_force_threshold)
    if args.excessive_force_threshold is not None:
        safety_cfg["excessive_force_threshold"] = float(args.excessive_force_threshold)
    if args.speed_threshold is not None:
        safety_cfg["speed_threshold"] = float(args.speed_threshold)
    if args.drop_velocity_threshold is not None:
        safety_cfg["drop_velocity_threshold"] = float(args.drop_velocity_threshold)

    bounds = {}
    if args.boundary_x is not None:
        bounds["x"] = [float(args.boundary_x[0]), float(args.boundary_x[1])]
    if args.boundary_y is not None:
        bounds["y"] = [float(args.boundary_y[0]), float(args.boundary_y[1])]
    if args.boundary_z is not None:
        bounds["z"] = [float(args.boundary_z[0]), float(args.boundary_z[1])]
    if bounds:
        safety_cfg["boundary_bounds"] = bounds

    labeler = SafetyLabeler({"safety": safety_cfg})
    return labeler.label_episode_arrays(
        {
            "eef_positions": tensors["eef_positions"].cpu().numpy(),
            "contact_forces": tensors["contact_forces"].cpu().numpy(),
            "actions": tensors["actions"].cpu().numpy(),
        }
    )


def success_trace(meta: dict, num_steps: int) -> list[bool]:
    series = [bool(x) for x in meta.get("success_by_timestep", [])]
    if len(series) == num_steps:
        return series
    first_success = meta.get("first_success_step", None)
    if first_success is not None and num_steps > 0:
        trace = [False] * num_steps
        for step_idx in range(int(first_success), num_steps):
            trace[step_idx] = True
        return trace
    return [False] * num_steps


def episode_row(meta: dict, temporal_meta: dict, categories: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "rollout_id": meta.get("subset_target_stem") or meta.get("rollout_id") or meta.get("timestamp"),
        "suite": meta.get("suite", ""),
        "task_idx": meta.get("task_idx", ""),
        "episode_success": bool(meta.get("episode_success", False)),
        "episode_failure": bool(meta.get("episode_failure", not bool(meta.get("episode_success", False)))),
        "has_violation": bool(meta.get("has_violations", False)),
        "has_violation_onset": bool(temporal_meta.get("total_violation_onsets", 0) > 0),
        "first_violation_step": meta.get("first_violation_step", None),
        "first_violation_onset_step": temporal_meta.get("first_violation_onset_step", None),
        "total_violations": int(meta.get("total_violations", 0)),
        "total_violation_onsets": int(temporal_meta.get("total_violation_onsets", 0)),
        "num_steps": int(meta.get("num_steps", 0)),
    }
    violation_counts = meta.get("violation_counts", {}) or {}
    onset_counts = temporal_meta.get("violation_onset_counts_by_category", {}) or {}
    for category in categories:
        row[f"{category}_active_steps"] = int(violation_counts.get(category, 0))
        row[f"{category}_onsets"] = int(onset_counts.get(category, 0))
    return row


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(args.output_dir)
    event_min_steps = parse_event_min_steps(list(args.event_min_steps))

    tensor_paths = sorted(data_dir.glob("rollout_*.safetensors"))
    if not tensor_paths:
        raise FileNotFoundError(f"No rollout tensors found in {data_dir}")

    rows: list[dict[str, object]] = []
    category_active_episode_counts = {category: 0 for category in SAFETY_CATEGORIES}
    category_onset_episode_counts = {category: 0 for category in SAFETY_CATEGORIES}
    category_active_step_totals = {category: 0 for category in SAFETY_CATEGORIES}
    category_onset_totals = {category: 0 for category in SAFETY_CATEGORIES}
    first_active_steps: list[int] = []
    first_onset_steps: list[int] = []
    success_count = 0
    violation_episode_count = 0
    onset_episode_count = 0

    for tensor_path in tensor_paths:
        meta_path = tensor_path.with_suffix(".json")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        categories = list(meta.get("safety_categories", SAFETY_CATEGORIES))
        tensors = load_file(str(tensor_path))
        labels_np = (
            relabel_from_tensors(tensors, args=args)
            if args.relabel_from_tensors
            else normalize_safety_labels(tensors, num_categories=len(categories))
        )
        num_steps = int(labels_np.shape[0])

        temporal_meta = build_temporal_event_metadata(
            success_flags=success_trace(meta, num_steps=num_steps),
            safety_matrix=labels_np,
            categories=categories,
            min_active_steps_by_category=event_min_steps or None,
        )

        meta["num_steps"] = int(meta.get("num_steps", num_steps))
        if "episode_success" not in meta:
            meta["episode_success"] = bool(tensors.get("episode_success", torch.zeros(1, dtype=torch.bool)).bool().any().item())
        meta["episode_failure"] = bool(meta.get("episode_failure", not bool(meta.get("episode_success", False))))
        meta["has_violations"] = bool(labels_np.any())
        meta["violation_counts"] = {
            category: int(labels_np[:, idx].sum()) if labels_np.shape[1] > idx else 0
            for idx, category in enumerate(categories)
        }
        meta["total_violations"] = int(labels_np.any(axis=1).sum()) if labels_np.size else 0
        meta.update(temporal_meta)

        if args.rewrite_metadata:
            save_json(meta_path, meta)

        success_count += int(bool(meta.get("episode_success", False)))
        violation_episode_count += int(bool(meta.get("has_violations", False)))
        onset_episode_count += int(int(meta.get("total_violation_onsets", 0)) > 0)
        if meta.get("first_violation_step", None) is not None:
            first_active_steps.append(int(meta["first_violation_step"]))
        if meta.get("first_violation_onset_step", None) is not None:
            first_onset_steps.append(int(meta["first_violation_onset_step"]))

        for category in categories:
            active_steps = int(meta["violation_counts"].get(category, 0))
            onset_count = int(meta.get("violation_onset_counts_by_category", {}).get(category, 0))
            category_active_step_totals[category] += active_steps
            category_onset_totals[category] += onset_count
            category_active_episode_counts[category] += int(active_steps > 0)
            category_onset_episode_counts[category] += int(onset_count > 0)

        rows.append(episode_row(meta, temporal_meta=temporal_meta, categories=categories))

    per_episode_path = output_dir / "rollout_label_audit_per_episode.csv"
    with per_episode_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    num_episodes = len(rows)
    summary = {
        "num_episodes": num_episodes,
        "success_episode_count": success_count,
        "success_episode_rate": float(success_count / max(num_episodes, 1)),
        "violation_episode_count": violation_episode_count,
        "violation_episode_rate": float(violation_episode_count / max(num_episodes, 1)),
        "violation_onset_episode_count": onset_episode_count,
        "violation_onset_episode_rate": float(onset_episode_count / max(num_episodes, 1)),
        "category_active_episode_counts": category_active_episode_counts,
        "category_active_episode_rates": {
            category: float(count / max(num_episodes, 1)) for category, count in category_active_episode_counts.items()
        },
        "category_onset_episode_counts": category_onset_episode_counts,
        "category_onset_episode_rates": {
            category: float(count / max(num_episodes, 1)) for category, count in category_onset_episode_counts.items()
        },
        "category_active_step_totals": category_active_step_totals,
        "category_onset_totals": category_onset_totals,
        "median_first_violation_step": float(np.median(first_active_steps)) if first_active_steps else None,
        "median_first_violation_onset_step": float(np.median(first_onset_steps)) if first_onset_steps else None,
        "event_min_active_steps_by_category": {
            **DEFAULT_EVENT_MIN_ACTIVE_STEPS,
            **event_min_steps,
        },
        "relabel_from_tensors": bool(args.relabel_from_tensors),
        "safety_threshold_overrides": {
            key: value
            for key, value in {
                "collision_force_threshold": args.collision_force_threshold,
                "excessive_force_threshold": args.excessive_force_threshold,
                "speed_threshold": args.speed_threshold,
                "drop_velocity_threshold": args.drop_velocity_threshold,
                "boundary_x": args.boundary_x,
                "boundary_y": args.boundary_y,
                "boundary_z": args.boundary_z,
            }.items()
            if value is not None
        },
        "metadata_rewritten": bool(args.rewrite_metadata),
    }
    save_json(output_dir / "rollout_label_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
