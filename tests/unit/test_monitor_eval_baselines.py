import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from safetensors.torch import save_file

from src.sae.model import BatchTopKSAE


def _write_rollout(path: Path, d_in: int, steps: int, unsafe: bool, task_idx: int = 0):
    acts = torch.randn(steps, 7, d_in, dtype=torch.float16)
    labels = torch.zeros(steps, 5, dtype=torch.bool)
    forces = torch.full((steps,), 5.0, dtype=torch.float32)
    if unsafe:
        labels[::2, 0] = True
        labels[::2, 1] = True
        forces[::2] = 60.0

    data = {
        "activations_layer16": acts,
        "safety_labels": labels,
        "episode_safety_violations": labels.sum(dim=0).to(torch.int32),
        "actions": torch.randn(steps, 7, dtype=torch.float32),
        "eef_positions": torch.randn(steps, 3, dtype=torch.float32),
        "contact_forces": forces,
        "episode_success": torch.tensor([not unsafe], dtype=torch.bool),
    }
    save_file(data, str(path))
    success_by_timestep = [False] * steps
    violation_by_timestep = [False] * steps
    violation_by_timestep_by_category = {
        "collision": [False] * steps,
        "excessive_force": [False] * steps,
        "boundary_violation": [False] * steps,
        "high_approach_speed": [False] * steps,
        "object_drop": [False] * steps,
    }
    if unsafe and steps >= 3:
        violation_by_timestep[2] = True
        violation_by_timestep_by_category["collision"][2] = True
        violation_by_timestep_by_category["excessive_force"][2] = True
    violation_onset_by_timestep = list(violation_by_timestep)
    violation_onset_by_timestep_by_category = {
        key: list(value) for key, value in violation_by_timestep_by_category.items()
    }

    meta = {
        "suite": "goal",
        "task_idx": task_idx,
        "num_steps": steps,
        "episode_success": bool(not unsafe),
        "episode_failure": bool(unsafe),
        "has_violations": bool(unsafe),
        "violation_counts": {
            "collision": int(unsafe),
            "excessive_force": int(unsafe),
            "boundary_violation": 0,
            "high_approach_speed": 0,
            "object_drop": 0,
        },
        "total_violations": int(unsafe),
        "success_by_timestep": success_by_timestep,
        "first_success_step": None,
        "violation_by_timestep": violation_by_timestep,
        "violation_onset_by_timestep": violation_onset_by_timestep,
        "first_violation_step": 2 if unsafe and steps >= 3 else None,
        "first_violation_onset_step": 2 if unsafe and steps >= 3 else None,
        "violation_by_timestep_by_category": violation_by_timestep_by_category,
        "violation_onset_by_timestep_by_category": violation_onset_by_timestep_by_category,
        "first_violation_step_by_category": {
            "collision": 2 if unsafe and steps >= 3 else None,
            "excessive_force": 2 if unsafe and steps >= 3 else None,
            "boundary_violation": None,
            "high_approach_speed": None,
            "object_drop": None,
        },
        "first_violation_onset_step_by_category": {
            "collision": 2 if unsafe and steps >= 3 else None,
            "excessive_force": 2 if unsafe and steps >= 3 else None,
            "boundary_violation": None,
            "high_approach_speed": None,
            "object_drop": None,
        },
        "violation_onset_counts_by_category": {
            "collision": int(unsafe),
            "excessive_force": int(unsafe),
            "boundary_violation": 0,
            "high_approach_speed": 0,
            "object_drop": 0,
        },
        "total_violation_onsets": int(unsafe),
    }
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def test_monitor_eval_outputs_all_baselines(tmp_path: Path):
    d_in = 16
    d_sae = 32
    k = 4

    data_dir = tmp_path / "rollouts"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(6):
        _write_rollout(data_dir / f"rollout_{i:06d}.safetensors", d_in=d_in, steps=6, unsafe=(i % 2 == 0))

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    ckpt_path = tmp_path / "sae.pt"
    torch.save({"model_state_dict": model.state_dict(), "d_in": d_in, "d_sae": d_sae, "k": k}, ckpt_path)

    sae_cfg = {
        "sae": {"d_in": d_in, "d_sae": d_sae, "k": k},
        "training": {"checkpoint_interval": 10},
    }
    sae_cfg_path = tmp_path / "sae.yaml"
    sae_cfg_path.write_text(yaml.safe_dump(sae_cfg), encoding="utf-8")

    eval_cfg = {
        "safety_analysis": {"test_split": 0.3},
        "monitor": {
            "pareto_threshold_step": 0.1,
            "latency_repeats": 16,
            "calibration_method": "platt",
            "calibration_split": 0.25,
            "threshold_selection_metric": "cost_weighted_f1",
            "threshold_grid_size": 21,
            "ece_num_bins": 10,
            "telemetry_window_steps": 4,
            "operating_point_false_alarm_budgets": [0.05, 0.10],
            "feature_export_top_k": 8,
        },
        "baselines": {
            "random": True,
            "force_threshold": True,
            "telemetry_lr": True,
            "raw_activation_lr": True,
            "raw_activation_mlp": True,
        },
    }
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")

    rollout_cfg = {"safety": {"excessive_force_threshold": 50.0}}
    rollout_cfg_path = tmp_path / "rollout.yaml"
    rollout_cfg_path.write_text(yaml.safe_dump(rollout_cfg), encoding="utf-8")

    out_dir = tmp_path / "results"
    env = _subprocess_env()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitor.evaluate_monitor",
            "--sae_checkpoint",
            str(ckpt_path),
            "--layer",
            "16",
            "--data_dir",
            str(data_dir),
            "--sae_config",
            str(sae_cfg_path),
            "--eval_config",
            str(eval_cfg_path),
            "--rollout_config",
            str(rollout_cfg_path),
            "--output_dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )

    mdf = pd.read_csv(out_dir / "layer16_monitor_metrics.csv")
    methods = set(mdf["method"].tolist())
    assert {
        "sae_lr",
        "sae_threshold",
        "telemetry_lr",
        "raw_activation_lr",
        "raw_activation_mlp",
        "random",
        "force_threshold",
    }.issubset(methods)
    assert "brier" in mdf.columns
    assert "ece" in mdf.columns
    assert "calibration_method" in mdf.columns
    assert "threshold_selection_metric" in mdf.columns
    assert "telemetry_window_steps" in mdf.columns

    assert (out_dir / "layer16_roc_points.csv").exists()
    assert (out_dir / "layer16_per_category_roc.csv").exists()
    assert (out_dir / "layer16_pareto.csv").exists()
    assert (out_dir / "layer16_latency_ms.csv").exists()
    assert (out_dir / "layer16_threshold_selection.csv").exists()
    assert (out_dir / "layer16_threshold_selection_by_split.csv").exists()
    assert (out_dir / "layer16_operating_points.csv").exists()
    assert (out_dir / "layer16_operating_points_by_split.csv").exists()
    assert (out_dir / "layer16_sae_feature_weights.csv").exists()
    assert (out_dir / "layer16_sae_feature_weights_by_split.csv").exists()

    feature_df = pd.read_csv(out_dir / "layer16_sae_feature_weights.csv")
    assert "feature_idx" in feature_df.columns
    assert "consensus_rank" in feature_df.columns


def test_monitor_eval_prefix_task_split_outputs(tmp_path: Path):
    d_in = 16
    d_sae = 32
    k = 4

    data_dir = tmp_path / "rollouts"
    data_dir.mkdir(parents=True, exist_ok=True)

    rollouts = [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    ]
    for i, (task_idx, unsafe) in enumerate(rollouts):
        _write_rollout(
            data_dir / f"rollout_{i:06d}.safetensors",
            d_in=d_in,
            steps=6,
            unsafe=unsafe,
            task_idx=task_idx,
        )

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    ckpt_path = tmp_path / "sae.pt"
    torch.save({"model_state_dict": model.state_dict(), "d_in": d_in, "d_sae": d_sae, "k": k}, ckpt_path)

    sae_cfg_path = tmp_path / "sae.yaml"
    sae_cfg_path.write_text(yaml.safe_dump({"sae": {"d_in": d_in, "d_sae": d_sae, "k": k}}), encoding="utf-8")

    eval_cfg = {
        "safety_analysis": {"test_split": 0.34},
        "monitor": {
            "evaluation_mode": "prefix",
            "split_mode": "task",
            "calibration_method": "platt",
            "calibration_split": 0.25,
            "threshold_selection_metric": "cost_weighted_f1",
            "threshold_grid_size": 21,
            "max_false_alarm_rate_success_episodes": 0.2,
            "telemetry_window_steps": 4,
            "operating_point_false_alarm_budgets": [0.05, 0.10],
            "feature_export_top_k": 8,
            "prediction_target": "violation",
            "future_horizon": 3,
            "min_prefix": 1,
            "prefix_stride": 2,
            "pareto_threshold_step": 0.2,
            "latency_repeats": 8,
        },
        "baselines": {
            "random": True,
            "force_threshold": True,
            "telemetry_lr": True,
            "raw_activation_lr": True,
            "raw_activation_mlp": True,
        },
    }
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")

    rollout_cfg_path = tmp_path / "rollout.yaml"
    rollout_cfg_path.write_text(yaml.safe_dump({"safety": {"excessive_force_threshold": 50.0}}), encoding="utf-8")

    out_dir = tmp_path / "results"
    env = _subprocess_env()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitor.evaluate_monitor",
            "--sae_checkpoint",
            str(ckpt_path),
            "--layer",
            "16",
            "--data_dir",
            str(data_dir),
            "--sae_config",
            str(sae_cfg_path),
            "--eval_config",
            str(eval_cfg_path),
            "--rollout_config",
            str(rollout_cfg_path),
            "--output_dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )

    mdf = pd.read_csv(out_dir / "layer16_monitor_metrics.csv")
    assert (mdf["evaluation_mode"] == "prefix").all()
    assert (mdf["split_mode"] == "task").all()
    assert (mdf["prediction_target"] == "violation").all()
    assert "mean_detection_lead_time" in mdf.columns
    assert "false_alarm_onsets_per_success_episode" in mdf.columns
    assert (mdf["calibration_method"] == "platt").all()
    assert "brier" in mdf.columns
    assert "ece" in mdf.columns
    assert "telemetry_window_steps" in mdf.columns
    assert "telemetry_lr" in set(mdf["method"].tolist())

    operating_df = pd.read_csv(out_dir / "layer16_operating_points.csv")
    assert set(operating_df["target_false_alarm_budget"].round(2).tolist()) == {0.05, 0.10}
    assert "telemetry_lr" in set(operating_df["method"].tolist())

    feature_df = pd.read_csv(out_dir / "layer16_sae_feature_weights.csv")
    assert feature_df["feature_idx"].is_unique

    cat_df = pd.read_csv(out_dir / "layer16_per_category_auroc.csv")
    assert (cat_df["evaluation_mode"] == "prefix").all()
    assert (cat_df["split_mode"] == "task").all()


def test_monitor_eval_prefix_onset_target_outputs(tmp_path: Path):
    d_in = 16
    d_sae = 32
    k = 4

    data_dir = tmp_path / "rollouts"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(6):
        _write_rollout(
            data_dir / f"rollout_{i:06d}.safetensors",
            d_in=d_in,
            steps=6,
            unsafe=(i % 2 == 0),
            task_idx=i // 2,
        )

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    ckpt_path = tmp_path / "sae.pt"
    torch.save({"model_state_dict": model.state_dict(), "d_in": d_in, "d_sae": d_sae, "k": k}, ckpt_path)

    sae_cfg_path = tmp_path / "sae.yaml"
    sae_cfg_path.write_text(yaml.safe_dump({"sae": {"d_in": d_in, "d_sae": d_sae, "k": k}}), encoding="utf-8")

    eval_cfg = {
        "safety_analysis": {"test_split": 0.34},
        "monitor": {
            "evaluation_mode": "prefix",
            "split_mode": "task",
            "calibration_method": "platt",
            "calibration_split": 0.25,
            "threshold_selection_metric": "cost_weighted_f1",
            "threshold_grid_size": 21,
            "telemetry_window_steps": 4,
            "operating_point_false_alarm_budgets": [0.05, 0.10],
            "prediction_target": "violation_onset",
            "future_horizon": 3,
            "min_prefix": 1,
            "prefix_stride": 2,
            "pareto_threshold_step": 0.2,
            "latency_repeats": 8,
        },
        "baselines": {
            "random": True,
            "force_threshold": True,
            "telemetry_lr": True,
            "raw_activation_lr": True,
            "raw_activation_mlp": True,
        },
    }
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")

    rollout_cfg_path = tmp_path / "rollout.yaml"
    rollout_cfg_path.write_text(yaml.safe_dump({"safety": {"excessive_force_threshold": 50.0}}), encoding="utf-8")

    out_dir = tmp_path / "results"
    env = _subprocess_env()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitor.evaluate_monitor",
            "--sae_checkpoint",
            str(ckpt_path),
            "--layer",
            "16",
            "--data_dir",
            str(data_dir),
            "--sae_config",
            str(sae_cfg_path),
            "--eval_config",
            str(eval_cfg_path),
            "--rollout_config",
            str(rollout_cfg_path),
            "--output_dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )

    mdf = pd.read_csv(out_dir / "layer16_monitor_metrics.csv")
    assert (mdf["prediction_target"] == "violation_onset").all()
    assert "detected_event_rate" in mdf.columns


def test_monitor_eval_prefix_leave_one_task_out_outputs(tmp_path: Path):
    d_in = 16
    d_sae = 32
    k = 4

    data_dir = tmp_path / "rollouts"
    data_dir.mkdir(parents=True, exist_ok=True)

    rollouts = []
    for task_idx in range(4):
        rollouts.append((task_idx, False))
        rollouts.append((task_idx, True))

    for i, (task_idx, unsafe) in enumerate(rollouts):
        _write_rollout(
            data_dir / f"rollout_{i:06d}.safetensors",
            d_in=d_in,
            steps=6,
            unsafe=unsafe,
            task_idx=task_idx,
        )

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    ckpt_path = tmp_path / "sae.pt"
    torch.save({"model_state_dict": model.state_dict(), "d_in": d_in, "d_sae": d_sae, "k": k}, ckpt_path)

    sae_cfg_path = tmp_path / "sae.yaml"
    sae_cfg_path.write_text(yaml.safe_dump({"sae": {"d_in": d_in, "d_sae": d_sae, "k": k}}), encoding="utf-8")

    eval_cfg = {
        "safety_analysis": {"test_split": 0.25},
        "monitor": {
            "evaluation_mode": "prefix",
            "split_mode": "task",
            "task_eval_mode": "leave_one_task_out",
            "calibration_method": "platt",
            "calibration_split": 0.25,
            "threshold_selection_metric": "cost_weighted_f1",
            "threshold_grid_size": 21,
            "max_false_alarm_rate_success_episodes": 0.2,
            "telemetry_window_steps": 4,
            "operating_point_false_alarm_budgets": [0.05, 0.10],
            "feature_export_top_k": 8,
            "prediction_target": "violation",
            "future_horizon": 3,
            "min_prefix": 1,
            "prefix_stride": 2,
            "pareto_threshold_step": 0.2,
            "latency_repeats": 8,
        },
        "baselines": {
            "random": True,
            "force_threshold": True,
            "telemetry_lr": True,
            "raw_activation_lr": True,
            "raw_activation_mlp": True,
        },
    }
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg), encoding="utf-8")

    rollout_cfg_path = tmp_path / "rollout.yaml"
    rollout_cfg_path.write_text(yaml.safe_dump({"safety": {"excessive_force_threshold": 50.0}}), encoding="utf-8")

    out_dir = tmp_path / "results"
    env = _subprocess_env()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitor.evaluate_monitor",
            "--sae_checkpoint",
            str(ckpt_path),
            "--layer",
            "16",
            "--data_dir",
            str(data_dir),
            "--sae_config",
            str(sae_cfg_path),
            "--eval_config",
            str(eval_cfg_path),
            "--rollout_config",
            str(rollout_cfg_path),
            "--output_dir",
            str(out_dir),
        ],
        check=True,
        env=env,
    )

    mdf = pd.read_csv(out_dir / "layer16_monitor_metrics.csv")
    assert (mdf["task_eval_mode"] == "leave_one_task_out").all()
    assert (mdf["num_splits"] == 4).all()
    assert (mdf["test_groups_per_split"] == 1).all()

    split_df = pd.read_csv(out_dir / "layer16_monitor_metrics_by_split.csv")
    assert split_df["split_id"].nunique() == 4
    assert (split_df["held_out_group_count"] == 1).all()
    assert (split_df["calibration_method"] == "platt").all()
    assert "telemetry_lr" in set(split_df["method"].tolist())
    assert (split_df["telemetry_window_steps"] == 4).all()

    cat_df = pd.read_csv(out_dir / "layer16_per_category_auroc.csv")
    assert (cat_df["task_eval_mode"] == "leave_one_task_out").all()

    assert (out_dir / "layer16_per_category_auroc_by_split.csv").exists()
    assert (out_dir / "layer16_pareto_by_split.csv").exists()
    assert (out_dir / "layer16_threshold_selection.csv").exists()
    assert (out_dir / "layer16_threshold_selection_by_split.csv").exists()
    assert (out_dir / "layer16_operating_points.csv").exists()
    assert (out_dir / "layer16_sae_feature_weights.csv").exists()

    operating_df = pd.read_csv(out_dir / "layer16_operating_points_by_split.csv")
    assert set(operating_df["target_false_alarm_budget"].round(2).tolist()) == {0.05, 0.10}
    assert operating_df["split_id"].nunique() == 4
