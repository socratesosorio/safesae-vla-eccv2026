"""Simulator-based causal validation for signed SAE feature interventions."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from src.sae.model import BatchTopKSAE
from src.sae.train_sae import BatchTopKSAE as LegacyBatchTopKSAE
from src.utils.config import load_yaml
from src.utils.metrics import bootstrap_rate_ci
from src.utils.runtime import ensure_dir

SAFETY_CATEGORY_TO_METRIC_KEY = {
    "collision": "collision",
    "excessive_force": "excessive_force",
    "boundary_violation": "boundary_violation",
    "high_approach_speed": "high_speed",
    "object_drop": "object_drop",
}


def _normalize_optional_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "feature_set"


def _parse_condition_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [name for name in (_normalize_optional_str(v) for v in value) if name]
    text = _normalize_optional_str(value)
    if not text:
        return []
    return [part for part in (_normalize_optional_str(x) for x in text.split(",")) if part]


def _parse_allowed_suites(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [suite for suite in (_normalize_optional_str(v) for v in value) if suite]
    text = _normalize_optional_str(value)
    if not text:
        return []
    return [suite for suite in (_normalize_optional_str(x) for x in text.split(",")) if suite]


def _parse_allowed_task_specs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = [_normalize_optional_str(v) for v in value]
    else:
        text = _normalize_optional_str(value)
        if not text:
            return []
        values = [_normalize_optional_str(x) for x in text.split(",")]
    out: list[str] = []
    for item in values:
        if not item:
            continue
        suite, sep, task_idx = item.partition(":")
        if not sep or not suite or not task_idx:
            raise ValueError(f"Invalid allowed task spec: {item!r}. Expected format '<suite>:<task_idx>'.")
        out.append(f"{suite}:{int(task_idx)}")
    return out


def _task_spec(entry: dict[str, Any]) -> str:
    return f"{str(entry.get('suite', ''))}:{int(entry.get('task_idx', -1))}"


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = _normalize_optional_str(value)
    if not text:
        return None
    return float(text)


def _coerce_optional_int(value: Any) -> int | None:
    coerced = _coerce_optional_float(value)
    if coerced is None:
        return None
    return int(coerced)


def _load_sae(path: str, sae_cfg: dict, device: torch.device):
    primary = sae_cfg.get("primary", sae_cfg.get("sae", sae_cfg))
    d_in = int(primary.get("d_in", 4096))
    d_sae = int(primary.get("d_sae", 16384))
    k = int(primary.get("k", 32))
    ckpt = torch.load(path, map_location=device)
    d_in = int(ckpt.get("d_in", d_in))
    d_sae = int(ckpt.get("d_sae", d_sae))
    k = int(ckpt.get("k", k))
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt

    for cls in (BatchTopKSAE, LegacyBatchTopKSAE):
        try:
            model = cls(d_in=d_in, d_sae=d_sae, k=k).to(device)  # type: ignore[call-arg]
            model.load_state_dict(state)
            model.eval()
            norm_factor = float(ckpt.get("norm_factor", 1.0))
            return model, norm_factor
        except Exception:
            continue
    raise RuntimeError(f"Unable to load SAE checkpoint from {path}")


def _noise_from_cfg(rollout_cfg: dict, rng: random.Random) -> float:
    noise_cfg = rollout_cfg.get("collection", {}).get("noise", {})
    if noise_cfg:
        p_clean = float(noise_cfg.get("clean_fraction", 0.60))
        p_mild = float(noise_cfg.get("mild_noise_fraction", 0.20))
        p_strong = float(noise_cfg.get("strong_noise_fraction", 0.20))
        x = rng.random()
        if x < p_clean:
            return 0.0
        if x < p_clean + p_mild:
            return float(noise_cfg.get("mild_noise_std", 0.03))
        if x < p_clean + p_mild + p_strong:
            return float(noise_cfg.get("strong_noise_std", 0.08))
        return 0.0
    frac = float(rollout_cfg.get("collection", {}).get("noise_fraction", 0.3))
    if rng.random() < frac:
        return float(rollout_cfg.get("collection", {}).get("noise_std", 0.05))
    return 0.0


def _sample_task(
    collector,
    rollout_cfg: dict,
    rng: random.Random,
    allowed_suites: list[str] | None = None,
) -> tuple[str, int, float]:
    suites = [str(suite) for suite in rollout_cfg["collection"]["per_suite"].keys()]
    if allowed_suites:
        allowed = set(allowed_suites)
        suites = [suite for suite in suites if suite in allowed]
    if not suites:
        raise RuntimeError("No suites remain after applying the requested causal suite filter")
    suite = suites[int(rng.random() * len(suites))]
    suite_obj = collector._suite_builder(suite)
    if hasattr(suite_obj, "get_num_tasks"):
        n_tasks = int(suite_obj.get_num_tasks())
    elif hasattr(suite_obj, "get_task_names"):
        n_tasks = len(suite_obj.get_task_names())
    else:
        n_tasks = max(int(rollout_cfg["collection"]["per_suite"].get(suite, 1)), 1)
    task_idx = rng.randint(0, max(0, n_tasks - 1))
    return suite, task_idx, _noise_from_cfg(rollout_cfg, rng)


def _default_schedule_entry(collector, suite: str, task_idx: int, noise_level: float) -> dict[str, Any]:
    try:
        return dict(collector._default_schedule_entry(suite=suite, task_idx=task_idx, add_noise=float(noise_level)))
    except TypeError:
        return dict(collector._default_schedule_entry(suite=suite, task_idx=task_idx, noise_level=float(noise_level)))


def _resolve_candidate_schedule(
    collector,
    *,
    hazard_category: str,
    condition_group: str,
    condition_names: list[str],
    allowed_suites: list[str],
    allowed_task_specs: list[str],
) -> list[dict[str, Any]]:
    if not hazard_category and not condition_group and not condition_names and not allowed_suites and not allowed_task_specs:
        return []
    if not hasattr(collector, "_build_schedule"):
        raise RuntimeError("Collector does not expose _build_schedule for targeted causal sampling")

    schedule = [dict(entry) for entry in collector._build_schedule()]
    condition_name_set = set(condition_names)
    filtered: list[dict[str, Any]] = []
    allowed_suite_set = set(allowed_suites)
    allowed_task_spec_set = set(allowed_task_specs)
    for entry in schedule:
        if allowed_suite_set and str(entry.get("suite")) not in allowed_suite_set:
            continue
        if allowed_task_spec_set and _task_spec(entry) not in allowed_task_spec_set:
            continue
        if hazard_category and _normalize_optional_str(entry.get("hazard_category")) != hazard_category:
            continue
        if condition_group and _normalize_optional_str(entry.get("condition_group")) != condition_group:
            continue
        if condition_name_set and _normalize_optional_str(entry.get("condition")) not in condition_name_set:
            continue
        filtered.append(entry)
    if not filtered:
        raise RuntimeError(
            "No collection-schedule entries matched the requested causal filters: "
            f"hazard_category={hazard_category or '<any>'}, "
            f"condition_group={condition_group or '<any>'}, "
            f"condition_names={condition_names or ['<any>']}, "
            f"allowed_suites={allowed_suites or ['<any>']}, "
            f"allowed_task_specs={allowed_task_specs or ['<any>']}"
        )
    return filtered


def _sample_schedule_entry(
    *,
    collector,
    rollout_cfg: dict,
    rng: random.Random,
    candidate_schedule: list[dict[str, Any]],
    allowed_suites: list[str],
) -> dict[str, Any]:
    if candidate_schedule:
        return dict(candidate_schedule[rng.randrange(len(candidate_schedule))])
    suite, task_idx, noise_level = _sample_task(collector, rollout_cfg, rng, allowed_suites=allowed_suites)
    return _default_schedule_entry(collector, suite=suite, task_idx=task_idx, noise_level=noise_level)


def _build_rollout_plan(
    *,
    collector,
    rollout_cfg: dict,
    num_rollouts: int,
    candidate_schedule: list[dict[str, Any]],
    allowed_suites: list[str],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for i in range(int(num_rollouts)):
        rng = random.Random(i)
        schedule_entry = _sample_schedule_entry(
            collector=collector,
            rollout_cfg=rollout_cfg,
            rng=rng,
            candidate_schedule=candidate_schedule,
            allowed_suites=allowed_suites,
        )
        schedule_entry = dict(schedule_entry)
        schedule_entry["_pair_seed"] = int(10_000 + i)
        schedule_entry["_plan_index"] = int(i)
        plan.append(schedule_entry)
    plan.sort(
        key=lambda entry: (
            str(entry.get("suite", "")),
            int(entry.get("task_idx", -1)),
            int(entry.get("_plan_index", -1)),
        )
    )
    return plan


def _collect_single_rollout(
    *,
    collector,
    is_pi0: bool,
    task_idx: int,
    suite: str,
    noise_level: float,
    schedule_entry: dict[str, Any],
) -> dict[str, Any]:
    if is_pi0:
        return collector.collect_single_rollout(
            task_idx=task_idx,
            suite=suite,
            noise_level=float(noise_level),
            schedule_entry=schedule_entry,
        )
    return collector.collect_single_rollout(
        task_idx=task_idx,
        suite=suite,
        add_noise=float(noise_level),
        schedule_entry=schedule_entry,
    )


def _rollout_metrics(rollout: dict) -> dict[str, Any]:
    labels = np.asarray(rollout["safety_labels"], dtype=bool)
    meta = dict(rollout.get("metadata", {}) or {})
    any_violation = bool(labels.any()) if labels.size else False
    success = bool(rollout.get("episode_success", False))
    return {
        "success": success,
        "clean_success": bool(success and not any_violation),
        "collision": bool(labels[:, 0].any()) if labels.size else False,
        "excessive_force": bool(labels[:, 1].any()) if labels.size else False,
        "boundary_violation": bool(labels[:, 2].any()) if labels.size else False,
        "high_speed": bool(labels[:, 3].any()) if labels.size else False,
        "object_drop": bool(labels[:, 4].any()) if labels.size else False,
        "any_violation": any_violation,
        "num_steps": int(labels.shape[0]) if labels.ndim == 2 else 0,
        "suite": _normalize_optional_str(meta.get("suite")),
        "task_idx": int(meta.get("task_idx", -1)),
        "condition": _normalize_optional_str(meta.get("collection_condition")),
        "condition_group": _normalize_optional_str(meta.get("collection_condition_group")),
        "hazard_category": _normalize_optional_str(meta.get("hazard_category")),
        "intervention_enabled": bool(meta.get("intervention_enabled", False)),
        "intervention_triggered": bool(meta.get("intervention_triggered", False)),
        "intervention_trigger_mode": _normalize_optional_str(meta.get("intervention_trigger_mode")),
        "intervention_trigger_signal_name": _normalize_optional_str(meta.get("intervention_trigger_signal_name")),
        "intervention_first_trigger_step": _coerce_optional_int(meta.get("intervention_first_trigger_step")),
        "intervention_last_trigger_step": _coerce_optional_int(meta.get("intervention_last_trigger_step")),
        "intervention_first_active_step": _coerce_optional_int(meta.get("intervention_first_active_step")),
        "intervention_last_active_step": _coerce_optional_int(meta.get("intervention_last_active_step")),
        "intervention_active_steps": int(meta.get("intervention_active_steps", 0) or 0),
        "intervention_trigger_eval_count": int(meta.get("intervention_trigger_eval_count", 0) or 0),
        "intervention_trigger_true_count": int(meta.get("intervention_trigger_true_count", 0) or 0),
        "intervention_trigger_true_fraction": float(meta.get("intervention_trigger_true_fraction", 0.0) or 0.0),
        "intervention_active_fraction": float(meta.get("intervention_active_fraction", 0.0) or 0.0),
        "intervention_mean_trigger_value": _coerce_optional_float(meta.get("intervention_mean_trigger_value")),
        "intervention_mean_trigger_margin": _coerce_optional_float(meta.get("intervention_mean_trigger_margin")),
        "intervention_max_trigger_value": _coerce_optional_float(meta.get("intervention_max_trigger_value")),
        "intervention_max_trigger_margin": _coerce_optional_float(meta.get("intervention_max_trigger_margin")),
    }


def _attach_plan_metadata(
    row: dict[str, Any],
    *,
    schedule_entry: dict[str, Any],
    pair_seed: int,
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["plan_index"] = int(schedule_entry.get("_plan_index", -1))
    enriched["pair_seed"] = int(pair_seed)
    enriched["schedule_idx"] = int(schedule_entry.get("schedule_idx", -1))
    return enriched


def _coerce_bool_sequence(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        seq = list(value)
    except TypeError:
        return None
    if not seq:
        return np.zeros(int(length), dtype=bool)
    out = np.zeros(int(length), dtype=bool)
    limit = min(int(length), len(seq))
    for idx in range(limit):
        out[idx] = bool(seq[idx])
    return out


def _compute_action_effect_metrics(baseline_rollout: dict[str, Any], clamped_rollout: dict[str, Any]) -> dict[str, Any]:
    base_actions = np.asarray(baseline_rollout.get("actions", []), dtype=np.float32)
    clamped_actions = np.asarray(clamped_rollout.get("actions", []), dtype=np.float32)
    if base_actions.ndim != 2 or clamped_actions.ndim != 2:
        return {}
    steps = int(min(base_actions.shape[0], clamped_actions.shape[0]))
    dims = int(min(base_actions.shape[1], clamped_actions.shape[1]))
    if steps <= 0 or dims <= 0:
        return {}

    delta = clamped_actions[:steps, :dims] - base_actions[:steps, :dims]
    delta_abs = np.abs(delta)
    step_l2 = np.linalg.norm(delta, axis=1)
    gripper_idx = min(6, dims - 1)

    def _mean_or_none(values: np.ndarray) -> float | None:
        return float(np.mean(values)) if values.size else None

    def _max_or_none(values: np.ndarray) -> float | None:
        return float(np.max(values)) if values.size else None

    translation_l2 = np.linalg.norm(delta[:, : min(3, dims)], axis=1) if dims >= 1 else np.asarray([], dtype=np.float32)
    rotation_l2 = (
        np.linalg.norm(delta[:, 3 : min(6, dims)], axis=1)
        if dims > 3
        else np.asarray([], dtype=np.float32)
    )
    gripper_abs = delta_abs[:, gripper_idx] if dims >= 1 else np.asarray([], dtype=np.float32)

    clamped_meta = dict(clamped_rollout.get("metadata", {}) or {})
    active_mask = _coerce_bool_sequence(
        clamped_meta.get("intervention_active_by_timestep"),
        length=steps,
    )
    if active_mask is None:
        active_mask = np.zeros(steps, dtype=bool)
    active_steps = int(active_mask.sum())

    active_delta = delta[active_mask] if active_steps else np.zeros((0, dims), dtype=np.float32)
    active_step_l2 = np.linalg.norm(active_delta, axis=1) if active_steps else np.asarray([], dtype=np.float32)
    active_translation_l2 = (
        np.linalg.norm(active_delta[:, : min(3, dims)], axis=1)
        if active_steps and dims >= 1
        else np.asarray([], dtype=np.float32)
    )
    active_rotation_l2 = (
        np.linalg.norm(active_delta[:, 3 : min(6, dims)], axis=1)
        if active_steps and dims > 3
        else np.asarray([], dtype=np.float32)
    )
    active_gripper_abs = (
        np.abs(active_delta[:, gripper_idx])
        if active_steps and dims >= 1
        else np.asarray([], dtype=np.float32)
    )

    return {
        "action_delta_steps_compared": int(steps),
        "action_delta_dims_compared": int(dims),
        "action_delta_any_nonzero": bool(np.any(delta_abs > 1e-6)),
        "action_delta_mean_l2": float(np.mean(step_l2)),
        "action_delta_max_l2": float(np.max(step_l2)),
        "action_delta_mean_abs": float(np.mean(delta_abs)),
        "action_delta_max_abs": float(np.max(delta_abs)),
        "action_delta_translation_mean_l2": _mean_or_none(translation_l2),
        "action_delta_translation_max_l2": _max_or_none(translation_l2),
        "action_delta_rotation_mean_l2": _mean_or_none(rotation_l2),
        "action_delta_rotation_max_l2": _max_or_none(rotation_l2),
        "action_delta_gripper_mean_abs": _mean_or_none(gripper_abs),
        "action_delta_gripper_max_abs": _max_or_none(gripper_abs),
        "action_delta_active_steps": int(active_steps),
        "action_delta_active_fraction": float(active_steps / max(steps, 1)),
        "action_delta_active_any_nonzero": bool(np.any(np.abs(active_delta) > 1e-6)) if active_steps else False,
        "action_delta_active_mean_l2": _mean_or_none(active_step_l2),
        "action_delta_active_max_l2": _max_or_none(active_step_l2),
        "action_delta_active_mean_abs": _mean_or_none(np.abs(active_delta)),
        "action_delta_active_max_abs": _max_or_none(np.abs(active_delta)),
        "action_delta_active_translation_mean_l2": _mean_or_none(active_translation_l2),
        "action_delta_active_translation_max_l2": _max_or_none(active_translation_l2),
        "action_delta_active_rotation_mean_l2": _mean_or_none(active_rotation_l2),
        "action_delta_active_rotation_max_l2": _max_or_none(active_rotation_l2),
        "action_delta_active_gripper_mean_abs": _mean_or_none(active_gripper_abs),
        "action_delta_active_gripper_max_abs": _max_or_none(active_gripper_abs),
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _rate_ci(key: str):
        events = [bool(r[key]) for r in rows]
        ci = bootstrap_rate_ci(events)
        return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi}

    def _mean(key: str, *, predicate=None):
        values = []
        for row in rows:
            if predicate is not None and not predicate(row):
                continue
            value = row.get(key, None)
            if value is None:
                continue
            values.append(float(value))
        return float(np.mean(values)) if values else None

    def _max(key: str, *, predicate=None):
        values = []
        for row in rows:
            if predicate is not None and not predicate(row):
                continue
            value = row.get(key, None)
            if value is None:
                continue
            values.append(float(value))
        return float(np.max(values)) if values else None

    signal_name = ""
    for row in rows:
        signal_name = _normalize_optional_str(row.get("intervention_trigger_signal_name"))
        if signal_name:
            break
    trigger_predicate = lambda row: bool(row.get("intervention_enabled", False))
    triggered_predicate = lambda row: bool(row.get("intervention_triggered", False))

    return {
        "collision_rate": _rate_ci("collision"),
        "excessive_force_rate": _rate_ci("excessive_force"),
        "boundary_violation_rate": _rate_ci("boundary_violation"),
        "high_speed_rate": _rate_ci("high_speed"),
        "object_drop_rate": _rate_ci("object_drop"),
        "any_violation_rate": _rate_ci("any_violation"),
        "success_rate": _rate_ci("success"),
        "clean_success_rate": _rate_ci("clean_success"),
        "mean_violations_per_episode": float(
            np.mean(
                [
                    int(r["collision"])
                    + int(r["excessive_force"])
                    + int(r["boundary_violation"])
                    + int(r["high_speed"])
                    + int(r["object_drop"])
                    for r in rows
                ]
            )
        )
        if rows
        else 0.0,
        "intervention_enabled_rate": _rate_ci("intervention_enabled"),
        "intervention_triggered_rate": _rate_ci("intervention_triggered"),
        "intervention_trigger_signal_name": signal_name,
        "mean_first_trigger_step": _mean("intervention_first_trigger_step", predicate=triggered_predicate),
        "mean_first_active_step": _mean("intervention_first_active_step", predicate=trigger_predicate),
        "mean_active_steps": _mean("intervention_active_steps", predicate=trigger_predicate),
        "mean_trigger_eval_count": _mean("intervention_trigger_eval_count", predicate=trigger_predicate),
        "mean_trigger_true_count": _mean("intervention_trigger_true_count", predicate=trigger_predicate),
        "mean_trigger_true_fraction": _mean("intervention_trigger_true_fraction", predicate=trigger_predicate),
        "mean_active_fraction": _mean("intervention_active_fraction", predicate=trigger_predicate),
        "mean_trigger_value": _mean("intervention_mean_trigger_value", predicate=trigger_predicate),
        "mean_trigger_margin": _mean("intervention_mean_trigger_margin", predicate=trigger_predicate),
        "max_trigger_value": _max("intervention_max_trigger_value", predicate=trigger_predicate),
        "max_trigger_margin": _max("intervention_max_trigger_margin", predicate=trigger_predicate),
        "mean_action_delta_steps_compared": _mean("action_delta_steps_compared"),
        "mean_action_delta_dims_compared": _mean("action_delta_dims_compared"),
        "action_delta_any_nonzero_rate": _mean("action_delta_any_nonzero"),
        "mean_action_delta_l2": _mean("action_delta_mean_l2"),
        "max_action_delta_l2": _max("action_delta_max_l2"),
        "mean_action_delta_abs": _mean("action_delta_mean_abs"),
        "max_action_delta_abs": _max("action_delta_max_abs"),
        "mean_action_delta_translation_l2": _mean("action_delta_translation_mean_l2"),
        "max_action_delta_translation_l2": _max("action_delta_translation_max_l2"),
        "mean_action_delta_rotation_l2": _mean("action_delta_rotation_mean_l2"),
        "max_action_delta_rotation_l2": _max("action_delta_rotation_max_l2"),
        "mean_action_delta_gripper_abs": _mean("action_delta_gripper_mean_abs"),
        "max_action_delta_gripper_abs": _max("action_delta_gripper_max_abs"),
        "mean_action_delta_active_steps": _mean("action_delta_active_steps"),
        "mean_action_delta_active_fraction": _mean("action_delta_active_fraction"),
        "action_delta_active_any_nonzero_rate": _mean("action_delta_active_any_nonzero"),
        "mean_action_delta_active_l2": _mean("action_delta_active_mean_l2"),
        "max_action_delta_active_l2": _max("action_delta_active_max_l2"),
        "mean_action_delta_active_abs": _mean("action_delta_active_mean_abs"),
        "max_action_delta_active_abs": _max("action_delta_active_max_abs"),
        "mean_action_delta_active_translation_l2": _mean("action_delta_active_translation_mean_l2"),
        "max_action_delta_active_translation_l2": _max("action_delta_active_translation_max_l2"),
        "mean_action_delta_active_rotation_l2": _mean("action_delta_active_rotation_mean_l2"),
        "max_action_delta_active_rotation_l2": _max("action_delta_active_rotation_max_l2"),
        "mean_action_delta_active_gripper_abs": _mean("action_delta_active_gripper_mean_abs"),
        "max_action_delta_active_gripper_abs": _max("action_delta_active_gripper_max_abs"),
        "per_episode": rows,
    }


def _paired_binary_p(clamped_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], key: str) -> float:
    clamped_values = np.asarray([int(x[key]) for x in clamped_rows], dtype=np.float32)
    baseline_values = np.asarray([int(x[key]) for x in baseline_rows], dtype=np.float32)
    try:
        return float(wilcoxon(clamped_values, baseline_values, zero_method="wilcox").pvalue)
    except Exception:
        return 1.0


def run_clamped_rollouts(
    config: dict,
    sae_path: str,
    feature_indices: list[int],
    num_rollouts: int = 100,
    scale: float = 0.0,
    output_path: str = "results/clamping_results.json",
    intervention_layer: int | None = None,
    sae_config_path: str = "configs/sae.yaml",
    feature_scale_map: dict[int, float] | None = None,
    feature_value_map: dict[int, float] | None = None,
    target_category: str = "",
    hazard_category: str = "",
    condition_group: str = "",
    condition_names: list[str] | None = None,
    allowed_suites: list[str] | None = None,
    allowed_task_specs: list[str] | None = None,
    feature_set: str = "selected_topk",
    selection_strategy: str = "",
    intervention_direction: str = "",
    is_random_control: bool = False,
    trigger_mode: str = "always",
    trigger_threshold: float | None = None,
    trigger_start_step: int = 0,
    trigger_end_step: int | None = None,
    trigger_latch: bool = True,
) -> dict[str, Any]:
    """Run paired baseline/clamped simulator rollouts under matched seeds and schedule entries."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_cfg = load_yaml(sae_config_path) if Path(sae_config_path).exists() else {}
    sae, norm_factor = _load_sae(sae_path, sae_cfg=sae_cfg, device=device)

    model_name = str(config.get("model", {}).get("name", "")).lower()
    d_in = int(config.get("activation_caching", {}).get("d_in", 4096))
    is_pi0 = ("pi0" in model_name) or (d_in == 2048)
    if is_pi0:
        from src.data.pi0_rollout_collector import Pi0RolloutCollector

        collector = Pi0RolloutCollector(config)
    else:
        from src.data.rollout_collector import RolloutCollector

        collector = RolloutCollector(config)

    acfg = config.get("activation_caching", {})
    if intervention_layer is not None:
        layer_idx = int(intervention_layer)
    elif "layer" in acfg:
        layer_idx = int(acfg.get("layer", 11 if is_pi0 else 20))
    else:
        default_layers = [9, 11, 14] if is_pi0 else [16, 20, 24]
        layers = [int(x) for x in acfg.get("layers", default_layers)]
        preferred = 11 if is_pi0 else 20
        layer_idx = preferred if preferred in layers else int(layers[0])

    target_category = _normalize_optional_str(target_category)
    hazard_category = _normalize_optional_str(hazard_category)
    condition_group = _normalize_optional_str(condition_group)
    normalized_condition_names = _parse_condition_names(condition_names)
    normalized_allowed_suites = _parse_allowed_suites(allowed_suites)
    normalized_allowed_task_specs = _parse_allowed_task_specs(allowed_task_specs)
    candidate_schedule = _resolve_candidate_schedule(
        collector,
        hazard_category=hazard_category,
        condition_group=condition_group,
        condition_names=normalized_condition_names,
        allowed_suites=normalized_allowed_suites,
        allowed_task_specs=normalized_allowed_task_specs,
    )
    rollout_plan = _build_rollout_plan(
        collector=collector,
        rollout_cfg=config,
        num_rollouts=int(num_rollouts),
        candidate_schedule=candidate_schedule,
        allowed_suites=normalized_allowed_suites,
    )

    clamped_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    partial_path = Path(output_path).with_suffix(".partial.json")
    start_idx = 0
    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as handle:
            partial = json.load(handle)
        clamped_rows = partial.get("clamped_rows", [])
        baseline_rows = partial.get("baseline_rows", [])
        start_idx = len(clamped_rows)

    for schedule_entry in tqdm(
        rollout_plan[start_idx:],
        desc=f"Paired causal rollouts ({feature_set})",
        initial=start_idx,
        total=int(num_rollouts),
    ):
        suite = str(schedule_entry["suite"])
        task_idx = int(schedule_entry["task_idx"])
        noise_level = float(schedule_entry.get("noise_level", 0.0))
        pair_seed = int(schedule_entry.get("_pair_seed", 10_000 + int(schedule_entry.get("_plan_index", 0))))

        if not is_pi0:
            checkpoint = collector.model_cfg["checkpoints"].get(suite, collector.model_name)
            collector._load_model(checkpoint)

        random.seed(pair_seed)
        np.random.seed(pair_seed)
        torch.manual_seed(pair_seed)
        base = _collect_single_rollout(
            collector=collector,
            is_pi0=is_pi0,
            task_idx=task_idx,
            suite=suite,
            noise_level=noise_level,
            schedule_entry=schedule_entry,
        )
        baseline_row = _attach_plan_metadata(
            _rollout_metrics(base),
            schedule_entry=schedule_entry,
            pair_seed=pair_seed,
        )
        baseline_rows.append(baseline_row)

        random.seed(pair_seed)
        np.random.seed(pair_seed)
        torch.manual_seed(pair_seed)
        if is_pi0:
            collector._load_policy()
            collector.activate_intervention(
                sae=sae,
                layer_idx=layer_idx,
                feature_indices=feature_indices,
                scale=float(scale),
                norm_factor=norm_factor,
                feature_scale_map=feature_scale_map,
                feature_value_map=feature_value_map,
                trigger_mode=str(trigger_mode),
                trigger_threshold=None if trigger_threshold is None else float(trigger_threshold),
                trigger_start_step=int(trigger_start_step),
                trigger_end_step=None if trigger_end_step is None else int(trigger_end_step),
                trigger_latch=bool(trigger_latch),
            )
            try:
                clamped = _collect_single_rollout(
                    collector=collector,
                    is_pi0=True,
                    task_idx=task_idx,
                    suite=suite,
                    noise_level=noise_level,
                    schedule_entry=schedule_entry,
                )
            finally:
                collector.deactivate_intervention()
        else:
            collector.activate_intervention(
                sae=sae,
                layer_idx=layer_idx,
                feature_indices=feature_indices,
                scale=float(scale),
                norm_factor=norm_factor,
                feature_scale_map=feature_scale_map,
                feature_value_map=feature_value_map,
                trigger_mode=str(trigger_mode),
                trigger_threshold=None if trigger_threshold is None else float(trigger_threshold),
                trigger_start_step=int(trigger_start_step),
                trigger_end_step=None if trigger_end_step is None else int(trigger_end_step),
                trigger_latch=bool(trigger_latch),
            )
            try:
                clamped = _collect_single_rollout(
                    collector=collector,
                    is_pi0=False,
                    task_idx=task_idx,
                    suite=suite,
                    noise_level=noise_level,
                    schedule_entry=schedule_entry,
                )
            finally:
                collector.deactivate_intervention()
        clamped_row = _attach_plan_metadata(
            _rollout_metrics(clamped),
            schedule_entry=schedule_entry,
            pair_seed=pair_seed,
        )
        clamped_row.update(_compute_action_effect_metrics(base, clamped))
        clamped_rows.append(clamped_row)

        completed = len(clamped_rows)
        # Flush after every completed pair so interrupted cluster jobs can resume
        # from the latest pair instead of losing up to 4 completed evaluations.
        if completed >= 1:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("w", encoding="utf-8") as handle:
                json.dump({"clamped_rows": clamped_rows, "baseline_rows": baseline_rows}, handle)

    if partial_path.exists():
        partial_path.unlink()

    clamped_metrics = _aggregate_metrics(clamped_rows)
    baseline_metrics = _aggregate_metrics(baseline_rows)
    target_metric_key = SAFETY_CATEGORY_TO_METRIC_KEY.get(target_category, "")

    paired_test = {
        "collision_wilcoxon_p": _paired_binary_p(clamped_rows, baseline_rows, "collision"),
        "any_violation_wilcoxon_p": _paired_binary_p(clamped_rows, baseline_rows, "any_violation"),
        "success_wilcoxon_p": _paired_binary_p(clamped_rows, baseline_rows, "success"),
        "clean_success_wilcoxon_p": _paired_binary_p(clamped_rows, baseline_rows, "clean_success"),
    }
    if target_metric_key:
        paired_test["target_category_wilcoxon_p"] = _paired_binary_p(clamped_rows, baseline_rows, target_metric_key)

    result = {
        "clamped": clamped_metrics,
        "baseline": baseline_metrics,
        "paired_test": paired_test,
        "config": {
            "feature_set": str(feature_set),
            "feature_indices": [int(x) for x in feature_indices],
            "feature_scale_map": {str(int(k)): float(v) for k, v in (feature_scale_map or {}).items()},
            "feature_value_map": {str(int(k)): float(v) for k, v in (feature_value_map or {}).items()},
            "scale": float(scale),
            "num_rollouts": int(num_rollouts),
            "layer": int(layer_idx),
            "target_category": target_category,
            "hazard_category": hazard_category,
            "condition_group": condition_group,
            "condition_names": normalized_condition_names,
            "allowed_suites": normalized_allowed_suites,
            "allowed_task_specs": normalized_allowed_task_specs,
            "sampling_pool_size": int(len(candidate_schedule)),
            "rollout_plan_grouped_by_suite": True,
            "selection_strategy": selection_strategy,
            "intervention_direction": intervention_direction,
            "is_random_control": bool(is_random_control),
            "trigger_mode": str(trigger_mode),
            "trigger_threshold": None if trigger_threshold is None else float(trigger_threshold),
            "trigger_start_step": int(trigger_start_step),
            "trigger_end_step": None if trigger_end_step is None else int(trigger_end_step),
            "trigger_latch": bool(trigger_latch),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def result_to_summary_row(
    *,
    result: dict[str, Any],
    feature_set: str,
    is_random_control: bool,
    scale: float,
    num_rollouts: int,
    num_features: int,
) -> dict[str, Any]:
    cfg = result.get("config", {})
    target_category = _normalize_optional_str(cfg.get("target_category"))
    target_metric_key = SAFETY_CATEGORY_TO_METRIC_KEY.get(target_category, "")
    feature_scale_map = cfg.get("feature_scale_map", {}) or {}
    scale_values = [float(v) for v in feature_scale_map.values()] if feature_scale_map else [float(scale)]

    row = {
        "feature_set": feature_set,
        "is_random_control": int(is_random_control),
        "num_features": int(num_features),
        "scale": float(scale),
        "feature_scale_min": float(min(scale_values)),
        "feature_scale_max": float(max(scale_values)),
        "feature_scale_mean": float(np.mean(scale_values)),
        "num_rollouts": int(num_rollouts),
        "target_category": target_category,
        "hazard_category": _normalize_optional_str(cfg.get("hazard_category")),
        "condition_group": _normalize_optional_str(cfg.get("condition_group")),
        "condition_names": ",".join(cfg.get("condition_names", [])),
        "allowed_suites": ",".join(cfg.get("allowed_suites", [])),
        "allowed_task_specs": ",".join(cfg.get("allowed_task_specs", [])),
        "selection_strategy": _normalize_optional_str(cfg.get("selection_strategy")),
        "intervention_direction": _normalize_optional_str(cfg.get("intervention_direction")),
        "sampling_pool_size": int(cfg.get("sampling_pool_size", 0)),
        "trigger_mode": _normalize_optional_str(cfg.get("trigger_mode")),
        "trigger_threshold": cfg.get("trigger_threshold", None),
        "trigger_start_step": int(cfg.get("trigger_start_step", 0)),
        "trigger_end_step": cfg.get("trigger_end_step", None),
        "trigger_latch": int(bool(cfg.get("trigger_latch", True))),
        "intervention_trigger_signal_name": result["clamped"].get("intervention_trigger_signal_name", ""),
        "intervention_triggered_rate_clamped": result["clamped"]["intervention_triggered_rate"]["mean"],
        "intervention_enabled_rate_clamped": result["clamped"]["intervention_enabled_rate"]["mean"],
        "mean_first_trigger_step_clamped": result["clamped"].get("mean_first_trigger_step", None),
        "mean_first_active_step_clamped": result["clamped"].get("mean_first_active_step", None),
        "mean_active_steps_clamped": result["clamped"].get("mean_active_steps", None),
        "mean_trigger_eval_count_clamped": result["clamped"].get("mean_trigger_eval_count", None),
        "mean_trigger_true_count_clamped": result["clamped"].get("mean_trigger_true_count", None),
        "mean_trigger_true_fraction_clamped": result["clamped"].get("mean_trigger_true_fraction", None),
        "mean_active_fraction_clamped": result["clamped"].get("mean_active_fraction", None),
        "mean_trigger_value_clamped": result["clamped"].get("mean_trigger_value", None),
        "mean_trigger_margin_clamped": result["clamped"].get("mean_trigger_margin", None),
        "max_trigger_value_clamped": result["clamped"].get("max_trigger_value", None),
        "max_trigger_margin_clamped": result["clamped"].get("max_trigger_margin", None),
        "action_delta_any_nonzero_rate_clamped": result["clamped"].get("action_delta_any_nonzero_rate", None),
        "mean_action_delta_steps_compared_clamped": result["clamped"].get("mean_action_delta_steps_compared", None),
        "mean_action_delta_l2_clamped": result["clamped"].get("mean_action_delta_l2", None),
        "max_action_delta_l2_clamped": result["clamped"].get("max_action_delta_l2", None),
        "mean_action_delta_abs_clamped": result["clamped"].get("mean_action_delta_abs", None),
        "mean_action_delta_translation_l2_clamped": result["clamped"].get("mean_action_delta_translation_l2", None),
        "mean_action_delta_rotation_l2_clamped": result["clamped"].get("mean_action_delta_rotation_l2", None),
        "mean_action_delta_gripper_abs_clamped": result["clamped"].get("mean_action_delta_gripper_abs", None),
        "mean_action_delta_active_steps_clamped": result["clamped"].get("mean_action_delta_active_steps", None),
        "mean_action_delta_active_fraction_clamped": result["clamped"].get("mean_action_delta_active_fraction", None),
        "action_delta_active_any_nonzero_rate_clamped": result["clamped"].get("action_delta_active_any_nonzero_rate", None),
        "mean_action_delta_active_l2_clamped": result["clamped"].get("mean_action_delta_active_l2", None),
        "max_action_delta_active_l2_clamped": result["clamped"].get("max_action_delta_active_l2", None),
        "mean_action_delta_active_abs_clamped": result["clamped"].get("mean_action_delta_active_abs", None),
        "mean_action_delta_active_translation_l2_clamped": result["clamped"].get("mean_action_delta_active_translation_l2", None),
        "mean_action_delta_active_rotation_l2_clamped": result["clamped"].get("mean_action_delta_active_rotation_l2", None),
        "mean_action_delta_active_gripper_abs_clamped": result["clamped"].get("mean_action_delta_active_gripper_abs", None),
        "collision_rate_clamped": result["clamped"]["collision_rate"]["mean"],
        "collision_rate_baseline": result["baseline"]["collision_rate"]["mean"],
        "success_rate_clamped": result["clamped"]["success_rate"]["mean"],
        "success_rate_baseline": result["baseline"]["success_rate"]["mean"],
        "clean_success_rate_clamped": result["clamped"]["clean_success_rate"]["mean"],
        "clean_success_rate_baseline": result["baseline"]["clean_success_rate"]["mean"],
        "any_violation_rate_clamped": result["clamped"]["any_violation_rate"]["mean"],
        "any_violation_rate_baseline": result["baseline"]["any_violation_rate"]["mean"],
        "collision_wilcoxon_p": result["paired_test"]["collision_wilcoxon_p"],
        "any_violation_wilcoxon_p": result["paired_test"]["any_violation_wilcoxon_p"],
        "success_wilcoxon_p": result["paired_test"]["success_wilcoxon_p"],
        "clean_success_wilcoxon_p": result["paired_test"]["clean_success_wilcoxon_p"],
    }
    if target_metric_key:
        metric_name = f"{target_metric_key}_rate"
        row["target_category_rate_clamped"] = result["clamped"][metric_name]["mean"]
        row["target_category_rate_baseline"] = result["baseline"][metric_name]["mean"]
        row["target_category_wilcoxon_p"] = result["paired_test"].get("target_category_wilcoxon_p", 1.0)
    return row


def _load_ranked_feature_sets(
    *,
    features_csv: str,
    top_k: int,
    scale: float,
    random_controls_csv: str,
) -> list[dict[str, Any]]:
    if not str(features_csv).strip():
        raise ValueError("features CSV is required when feature_manifest_csv is not provided")

    ranked = pd.read_csv(features_csv)
    feature_indices = ranked["feature_idx"].astype(int).head(int(top_k)).tolist()
    feature_sets: list[dict[str, Any]] = [
        {
            "feature_set": "selected_topk",
            "feature_indices": feature_indices,
            "feature_scale_map": {},
            "scale": float(scale),
            "target_category": "",
            "hazard_category": "",
            "condition_group": "",
            "condition_names": [],
            "allowed_suites": [],
            "allowed_task_specs": [],
            "selection_strategy": "monitor_topk",
            "intervention_direction": "suppress",
            "is_random_control": False,
            "trigger_mode": "always",
            "trigger_threshold": None,
            "trigger_start_step": 0,
            "trigger_end_step": None,
            "trigger_latch": True,
        }
    ]

    random_controls_csv = str(random_controls_csv).strip()
    if random_controls_csv:
        random_df = pd.read_csv(random_controls_csv)
        if "feature_set" not in random_df.columns or "feature_idx" not in random_df.columns:
            raise ValueError(f"{random_controls_csv} must contain feature_set and feature_idx columns")
        for feature_set, group in random_df.groupby("feature_set", sort=False):
            ordered = group.sort_values("rank") if "rank" in group.columns else group
            feature_sets.append(
                {
                    "feature_set": str(feature_set),
                    "feature_indices": ordered["feature_idx"].astype(int).tolist(),
                    "feature_scale_map": {},
                    "scale": float(scale),
                    "target_category": "",
                    "hazard_category": "",
                    "condition_group": "",
                    "condition_names": [],
                    "allowed_suites": [],
                    "allowed_task_specs": [],
                    "selection_strategy": "random_control",
                    "intervention_direction": "suppress",
                    "is_random_control": True,
                    "trigger_mode": "always",
                    "trigger_threshold": None,
                    "trigger_start_step": 0,
                    "trigger_end_step": None,
                    "trigger_latch": True,
                }
            )
    return feature_sets


def _load_manifest_feature_sets(feature_manifest_csv: str, default_scale: float) -> list[dict[str, Any]]:
    manifest_df = pd.read_csv(feature_manifest_csv)
    if "feature_set" not in manifest_df.columns or "feature_idx" not in manifest_df.columns:
        raise ValueError(f"{feature_manifest_csv} must contain feature_set and feature_idx columns")

    feature_sets: list[dict[str, Any]] = []
    for feature_set, group in manifest_df.groupby("feature_set", sort=False):
        ordered = group.copy()
        if "rank" in ordered.columns:
            ordered = ordered.sort_values(["rank", "feature_idx"], kind="stable")
        ordered = ordered.reset_index(drop=True)

        feature_scale_map: dict[int, float] = {}
        if "feature_scale" in ordered.columns:
            for row in ordered.itertuples(index=False):
                feat_scale = getattr(row, "feature_scale", np.nan)
                if pd.notna(feat_scale):
                    feature_scale_map[int(getattr(row, "feature_idx"))] = float(feat_scale)
        feature_value_map: dict[int, float] = {}
        if "feature_value" in ordered.columns:
            for row in ordered.itertuples(index=False):
                feat_value = getattr(row, "feature_value", np.nan)
                if pd.notna(feat_value):
                    feature_value_map[int(getattr(row, "feature_idx"))] = float(feat_value)

        condition_names = []
        if "condition_names" in ordered.columns:
            condition_names = _parse_condition_names(
                ordered["condition_names"].dropna().iloc[0] if not ordered["condition_names"].dropna().empty else ""
            )
        condition_col = "condition_name" if "condition_name" in ordered.columns else "condition"
        if not condition_names and condition_col in ordered.columns:
            condition_names = [
                name
                for name in (_normalize_optional_str(v) for v in ordered[condition_col].tolist())
                if name
            ]
            condition_names = list(dict.fromkeys(condition_names))

        scale = float(default_scale)
        if "default_scale" in ordered.columns:
            valid_scale = ordered["default_scale"].dropna()
            if not valid_scale.empty:
                scale = float(valid_scale.iloc[0])

        feature_sets.append(
            {
                "feature_set": str(feature_set),
                "feature_indices": ordered["feature_idx"].astype(int).tolist(),
                "feature_scale_map": feature_scale_map,
                "feature_value_map": feature_value_map,
                "scale": scale,
                "target_category": _normalize_optional_str(ordered.get("target_category", pd.Series(dtype=object)).dropna().iloc[0] if "target_category" in ordered.columns and not ordered["target_category"].dropna().empty else ""),
                "hazard_category": _normalize_optional_str(ordered.get("hazard_category", pd.Series(dtype=object)).dropna().iloc[0] if "hazard_category" in ordered.columns and not ordered["hazard_category"].dropna().empty else ""),
                "condition_group": _normalize_optional_str(ordered.get("condition_group", pd.Series(dtype=object)).dropna().iloc[0] if "condition_group" in ordered.columns and not ordered["condition_group"].dropna().empty else ""),
                "condition_names": condition_names,
                "allowed_suites": _parse_allowed_suites(
                    ordered.get("allowed_suites", pd.Series(dtype=object)).dropna().iloc[0]
                    if "allowed_suites" in ordered.columns and not ordered["allowed_suites"].dropna().empty
                    else ""
                ),
                "allowed_task_specs": _parse_allowed_task_specs(
                    ordered.get("allowed_task_specs", pd.Series(dtype=object)).dropna().iloc[0]
                    if "allowed_task_specs" in ordered.columns and not ordered["allowed_task_specs"].dropna().empty
                    else ""
                ),
                "selection_strategy": _normalize_optional_str(ordered.get("selection_strategy", pd.Series(dtype=object)).dropna().iloc[0] if "selection_strategy" in ordered.columns and not ordered["selection_strategy"].dropna().empty else ""),
                "intervention_direction": _normalize_optional_str(ordered.get("intervention_direction", pd.Series(dtype=object)).dropna().iloc[0] if "intervention_direction" in ordered.columns and not ordered["intervention_direction"].dropna().empty else ""),
                "is_random_control": bool(
                    int(ordered["is_random_control"].iloc[0]) if "is_random_control" in ordered.columns else str(feature_set).startswith("random_control")
                ),
                "trigger_mode": _normalize_optional_str(
                    ordered.get("trigger_mode", pd.Series(dtype=object)).dropna().iloc[0]
                    if "trigger_mode" in ordered.columns and not ordered["trigger_mode"].dropna().empty
                    else "always"
                )
                or "always",
                "trigger_threshold": (
                    float(ordered["trigger_threshold"].dropna().iloc[0])
                    if "trigger_threshold" in ordered.columns and not ordered["trigger_threshold"].dropna().empty
                    else None
                ),
                "trigger_start_step": int(
                    ordered["trigger_start_step"].dropna().iloc[0]
                    if "trigger_start_step" in ordered.columns and not ordered["trigger_start_step"].dropna().empty
                    else 0
                ),
                "trigger_end_step": (
                    int(ordered["trigger_end_step"].dropna().iloc[0])
                    if "trigger_end_step" in ordered.columns and not ordered["trigger_end_step"].dropna().empty
                    else None
                ),
                "trigger_latch": bool(
                    int(ordered["trigger_latch"].dropna().iloc[0])
                    if "trigger_latch" in ordered.columns and not ordered["trigger_latch"].dropna().empty
                    else 1
                ),
            }
        )
    return feature_sets


def load_feature_sets(
    *,
    features_csv: str,
    top_k: int,
    scale: float,
    random_controls_csv: str,
    feature_manifest_csv: str,
) -> list[dict[str, Any]]:
    manifest_path = _normalize_optional_str(feature_manifest_csv)
    if manifest_path:
        return _load_manifest_feature_sets(manifest_path, default_scale=float(scale))
    return _load_ranked_feature_sets(
        features_csv=features_csv,
        top_k=top_k,
        scale=float(scale),
        random_controls_csv=random_controls_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulator causal validation with signed feature interventions")
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--sae_checkpoint", type=str, required=True)
    parser.add_argument("--features", type=str, default="", help="CSV with feature_idx column")
    parser.add_argument("--feature_manifest_csv", type=str, default="")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/causal")
    parser.add_argument("--output_name", type=str, default="clamping_results.json")
    parser.add_argument("--random_controls_csv", type=str, default="")
    parser.add_argument("--target_category", type=str, default="")
    parser.add_argument("--hazard_category", type=str, default="")
    parser.add_argument("--condition_group", type=str, default="")
    parser.add_argument("--condition_names", type=str, default="")
    parser.add_argument("--allowed_suites", type=str, default="")
    parser.add_argument("--allowed_task_specs", type=str, default="")
    parser.add_argument("--trigger_mode", type=str, default="")
    parser.add_argument("--trigger_threshold", type=float, default=float("nan"))
    parser.add_argument("--trigger_start_step", type=int, default=-1)
    parser.add_argument("--trigger_end_step", type=int, default=-1)
    parser.add_argument("--trigger_latch", type=int, default=-1)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_cfg = load_yaml(args.rollout_config)
    eval_cfg = load_yaml(args.eval_config) if Path(args.eval_config).exists() else {}
    sim_cfg = eval_cfg.get("simulator_validation", {})

    default_target_category = _normalize_optional_str(args.target_category or sim_cfg.get("target_category", ""))
    default_hazard_category = _normalize_optional_str(args.hazard_category or sim_cfg.get("hazard_category", ""))
    default_condition_group = _normalize_optional_str(args.condition_group or sim_cfg.get("condition_group", ""))
    default_condition_names = _parse_condition_names(
        args.condition_names if _normalize_optional_str(args.condition_names) else sim_cfg.get("condition_names", [])
    )
    default_allowed_suites = _parse_allowed_suites(
        args.allowed_suites if _normalize_optional_str(args.allowed_suites) else sim_cfg.get("allowed_suites", [])
    )
    default_allowed_task_specs = _parse_allowed_task_specs(
        args.allowed_task_specs if _normalize_optional_str(args.allowed_task_specs) else sim_cfg.get("allowed_task_specs", [])
    )
    default_trigger_mode = _normalize_optional_str(args.trigger_mode or sim_cfg.get("trigger_mode", "always")) or "always"
    default_trigger_threshold = (
        float(args.trigger_threshold)
        if args.trigger_threshold == args.trigger_threshold
        else (None if sim_cfg.get("trigger_threshold", None) is None else float(sim_cfg.get("trigger_threshold")))
    )
    default_trigger_start_step = int(args.trigger_start_step if args.trigger_start_step >= 0 else sim_cfg.get("trigger_start_step", 0))
    default_trigger_end_step = None if args.trigger_end_step < 0 else int(args.trigger_end_step)
    if args.trigger_end_step < 0 and sim_cfg.get("trigger_end_step", None) is not None:
        default_trigger_end_step = int(sim_cfg.get("trigger_end_step"))
    default_trigger_latch = bool(int(args.trigger_latch)) if args.trigger_latch >= 0 else bool(sim_cfg.get("trigger_latch", True))

    feature_sets = load_feature_sets(
        features_csv=args.features,
        top_k=int(args.top_k),
        scale=float(args.scale),
        random_controls_csv=args.random_controls_csv,
        feature_manifest_csv=args.feature_manifest_csv,
    )

    out_dir = ensure_dir(args.output_dir)
    out_path = Path(out_dir) / args.output_name
    multi_set_mode = len(feature_sets) > 1 or bool(_normalize_optional_str(args.feature_manifest_csv))

    summary_rows: list[dict[str, Any]] = []
    controls_payload: dict[str, Any] = {}
    primary_result: dict[str, Any] | None = None
    primary_row: dict[str, Any] | None = None

    def _flush_summary_outputs() -> None:
        if primary_row is not None:
            pd.DataFrame([primary_row]).to_csv(Path(out_dir) / f"layer{args.layer}_causal_validation.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(Path(out_dir) / f"layer{args.layer}_causal_validation_controls.csv", index=False)
        with (Path(out_dir) / f"{out_path.stem}_controls.json").open("w", encoding="utf-8") as handle:
            json.dump(controls_payload, handle, indent=2)

    for idx, feature_set in enumerate(feature_sets):
        feature_name = str(feature_set["feature_set"])
        target_category = _normalize_optional_str(feature_set.get("target_category")) or default_target_category
        hazard_category = (
            _normalize_optional_str(feature_set.get("hazard_category"))
            or default_hazard_category
            or target_category
        )
        condition_group = _normalize_optional_str(feature_set.get("condition_group")) or default_condition_group
        condition_names = feature_set.get("condition_names") or default_condition_names
        allowed_suites = feature_set.get("allowed_suites") or default_allowed_suites
        allowed_task_specs = feature_set.get("allowed_task_specs") or default_allowed_task_specs
        trigger_mode = _normalize_optional_str(feature_set.get("trigger_mode")) or default_trigger_mode
        trigger_threshold = feature_set.get("trigger_threshold", default_trigger_threshold)
        resolved_trigger_threshold = _coerce_optional_float(trigger_threshold)
        trigger_start_step = int(feature_set.get("trigger_start_step", default_trigger_start_step))
        trigger_end_step = feature_set.get("trigger_end_step", default_trigger_end_step)
        resolved_trigger_end_step = _coerce_optional_int(trigger_end_step)
        trigger_latch = bool(feature_set.get("trigger_latch", default_trigger_latch))

        feature_json = out_path
        if multi_set_mode:
            feature_json = Path(out_dir) / f"{out_path.stem}_{_sanitize_name(feature_name)}.json"

        if multi_set_mode and feature_json.exists():
            with feature_json.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = run_clamped_rollouts(
                config=rollout_cfg,
                sae_path=args.sae_checkpoint,
                feature_indices=list(feature_set["feature_indices"]),
                num_rollouts=int(args.num_rollouts),
                scale=float(feature_set.get("scale", args.scale)),
                output_path=str(feature_json),
                intervention_layer=int(args.layer),
                sae_config_path=args.sae_config,
                feature_scale_map=dict(feature_set.get("feature_scale_map", {})),
                feature_value_map=dict(feature_set.get("feature_value_map", {})),
                target_category=target_category,
                hazard_category=hazard_category,
                condition_group=condition_group,
                condition_names=list(condition_names),
                allowed_suites=list(allowed_suites),
                allowed_task_specs=list(allowed_task_specs),
                feature_set=feature_name,
                selection_strategy=_normalize_optional_str(feature_set.get("selection_strategy")),
                intervention_direction=_normalize_optional_str(feature_set.get("intervention_direction")),
                is_random_control=bool(feature_set.get("is_random_control", False)),
                trigger_mode=trigger_mode,
                trigger_threshold=resolved_trigger_threshold,
                trigger_start_step=trigger_start_step,
                trigger_end_step=resolved_trigger_end_step,
                trigger_latch=trigger_latch,
            )

        row = result_to_summary_row(
            result=result,
            feature_set=feature_name,
            is_random_control=bool(feature_set.get("is_random_control", False)),
            scale=float(feature_set.get("scale", args.scale)),
            num_rollouts=int(args.num_rollouts),
            num_features=len(feature_set["feature_indices"]),
        )
        summary_rows.append(row)
        controls_payload[feature_name] = result

        if idx == 0:
            primary_result = result
            primary_row = row
        _flush_summary_outputs()

    if primary_result is None or primary_row is None:
        raise RuntimeError("No causal feature sets were evaluated")

    if not multi_set_mode:
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(primary_result, handle, indent=2)
    _flush_summary_outputs()


if __name__ == "__main__":
    main()
