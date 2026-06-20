"""Summarize closed-loop class-mean intervention runs.

The causal-validation JSONs contain paired baseline/clamped per-episode rows for
each feature set. This script computes rebuttal-ready summaries: success and
violation deltas, bootstrap confidence intervals, per-suite/task breakdowns, and
which action axes move under the intervention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--pattern", type=str, default="*.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def _mean_rate(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("mean", np.nan))
    return float(value)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_rows(data: dict[str, Any]) -> pd.DataFrame:
    baseline = data.get("baseline", {}).get("per_episode", [])
    clamped = data.get("clamped", {}).get("per_episode", [])
    by_plan = {int(row.get("plan_index", i)): dict(row) for i, row in enumerate(baseline)}
    rows: list[dict[str, Any]] = []
    for i, clamped_row in enumerate(clamped):
        plan_idx = int(clamped_row.get("plan_index", i))
        base_row = by_plan.get(plan_idx, {})
        row: dict[str, Any] = {
            "feature_set": str(data.get("config", {}).get("feature_set", "")),
            "is_random_control": bool(data.get("config", {}).get("is_random_control", False)),
            "suite": str(clamped_row.get("suite") or base_row.get("suite") or ""),
            "task_idx": int(clamped_row.get("task_idx", base_row.get("task_idx", -1))),
            "condition": str(clamped_row.get("condition") or base_row.get("condition") or ""),
            "plan_index": plan_idx,
            "baseline_success": int(bool(base_row.get("success", False))),
            "clamped_success": int(bool(clamped_row.get("success", False))),
            "baseline_any_violation": int(bool(base_row.get("any_violation", False))),
            "clamped_any_violation": int(bool(clamped_row.get("any_violation", False))),
            "baseline_collision": int(bool(base_row.get("collision", False))),
            "clamped_collision": int(bool(clamped_row.get("collision", False))),
        }
        for key in [
            "action_delta_active_mean_l2",
            "action_delta_active_translation_mean_l2",
            "action_delta_active_rotation_mean_l2",
            "action_delta_active_gripper_mean_abs",
            "action_delta_active_fraction",
        ]:
            row[key] = clamped_row.get(key, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["success_delta"] = df["clamped_success"] - df["baseline_success"]
        df["any_violation_delta"] = df["clamped_any_violation"] - df["baseline_any_violation"]
        df["collision_delta"] = df["clamped_collision"] - df["baseline_collision"]
    return df


def _summary_for_group(df: pd.DataFrame, *, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": int(len(df)),
        "baseline_success_rate": float(df["baseline_success"].mean()) if len(df) else float("nan"),
        "clamped_success_rate": float(df["clamped_success"].mean()) if len(df) else float("nan"),
        "success_delta": float(df["success_delta"].mean()) if len(df) else float("nan"),
        "baseline_any_violation_rate": float(df["baseline_any_violation"].mean()) if len(df) else float("nan"),
        "clamped_any_violation_rate": float(df["clamped_any_violation"].mean()) if len(df) else float("nan"),
        "any_violation_delta": float(df["any_violation_delta"].mean()) if len(df) else float("nan"),
        "mean_action_delta_active_l2": float(df["action_delta_active_mean_l2"].mean()) if len(df) else float("nan"),
        "mean_action_delta_active_translation_l2": float(df["action_delta_active_translation_mean_l2"].mean()) if len(df) else float("nan"),
        "mean_action_delta_active_rotation_l2": float(df["action_delta_active_rotation_mean_l2"].mean()) if len(df) else float("nan"),
        "mean_action_delta_active_gripper_abs": float(df["action_delta_active_gripper_mean_abs"].mean()) if len(df) else float("nan"),
    }
    for key in ["success_delta", "any_violation_delta", "action_delta_active_mean_l2"]:
        lo, hi = _bootstrap_ci(df[key].to_numpy(dtype=np.float32), n_boot=n_boot, rng=rng)
        out[f"{key}_ci95_low"] = lo
        out[f"{key}_ci95_high"] = hi
    return out


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    frames: list[pd.DataFrame] = []
    raw_rows: list[dict[str, Any]] = []
    paths = result_dir.rglob(args.pattern) if args.recursive else result_dir.glob(args.pattern)
    for path in sorted(paths):
        if path.name.endswith("_controls.json") or path.name.endswith("_summary.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "baseline" not in data or "clamped" not in data:
            continue
        df = _paired_rows(data)
        if df.empty:
            continue
        frames.append(df)
        cfg = data.get("config", {})
        raw_rows.append(
            {
                "feature_set": str(cfg.get("feature_set", path.stem)),
                "is_random_control": bool(cfg.get("is_random_control", False)),
                "success_wilcoxon_p": data.get("paired_test", {}).get("success_wilcoxon_p"),
                "any_violation_wilcoxon_p": data.get("paired_test", {}).get("any_violation_wilcoxon_p"),
                "result_file": str(path),
            }
        )

    all_pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_pairs.to_csv(output_dir / "closed_loop_intervention_pairs.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    if not all_pairs.empty:
        for feature_set, group in all_pairs.groupby("feature_set", sort=False):
            row = {"feature_set": feature_set, "is_random_control": bool(group["is_random_control"].iloc[0])}
            row.update(_summary_for_group(group, rng=rng, n_boot=int(args.bootstrap)))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if raw_rows and not summary.empty:
        raw = pd.DataFrame(raw_rows)
        summary = summary.merge(raw, on=["feature_set", "is_random_control"], how="left")
    summary.to_csv(output_dir / "closed_loop_intervention_summary.csv", index=False)

    breakdown_rows: list[dict[str, Any]] = []
    if not all_pairs.empty:
        for keys, group in all_pairs.groupby(["feature_set", "suite", "task_idx"], sort=False):
            feature_set, suite, task_idx = keys
            row = {"feature_set": feature_set, "suite": suite, "task_idx": int(task_idx)}
            row.update(_summary_for_group(group, rng=rng, n_boot=int(args.bootstrap)))
            breakdown_rows.append(row)
    pd.DataFrame(breakdown_rows).to_csv(output_dir / "closed_loop_intervention_task_breakdown.csv", index=False)

    payload = {
        "summary": summary.to_dict(orient="records"),
        "task_breakdown": breakdown_rows,
    }
    (output_dir / "closed_loop_intervention_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
