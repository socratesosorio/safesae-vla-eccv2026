from pathlib import Path

from safetensors.torch import save_file
import torch

from src.data.benchmark_repair import (
    audit_success_signals,
    build_repaired_benchmark_config,
    calibrate_safety_thresholds,
    screen_causal_slices,
    screen_tasks,
)
from src.utils.runtime import save_json


def _write_rollout(
    out_dir: Path,
    rollout_id: str,
    *,
    suite: str,
    task_idx: int,
    condition: str,
    success: bool,
    has_violations: bool,
    first_violation_step: int | None,
    first_violation_onset_step: int | None,
    eef_positions: list[list[float]],
    contact_forces: list[float],
    actions: list[list[float]],
    rewards: list[float] | None = None,
    episode_success_by_info: bool | None = None,
    episode_success_by_env_check: bool | None = None,
) -> None:
    tensor_path = out_dir / f"{rollout_id}.safetensors"
    payload = {
        "eef_positions": torch.tensor(eef_positions, dtype=torch.float32),
        "contact_forces": torch.tensor(contact_forces, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.float32),
    }
    if rewards is not None:
        payload["rewards"] = torch.tensor(rewards, dtype=torch.float32)
    save_file(payload, str(tensor_path))
    save_json(
        tensor_path.with_suffix(".json"),
        {
            "rollout_id": rollout_id,
            "suite": suite,
            "task_idx": task_idx,
            "task_name": f"{suite}_{task_idx}",
            "collection_condition": condition,
            "episode_success": success,
            "episode_success_by_info": success if episode_success_by_info is None else episode_success_by_info,
            "episode_success_by_env_check": success if episode_success_by_env_check is None else episode_success_by_env_check,
            "has_violations": has_violations,
            "first_violation_step": first_violation_step,
            "first_violation_onset_step": first_violation_onset_step,
            "total_violation_onsets": int(first_violation_onset_step is not None),
            "num_steps": len(eef_positions),
        },
    )


def test_calibrate_safety_thresholds_emits_per_suite_overrides(tmp_path: Path):
    data_dir = tmp_path / "rollouts"
    data_dir.mkdir()
    _write_rollout(
        data_dir,
        "rollout_000000",
        suite="object",
        task_idx=0,
        condition="clean",
        success=True,
        has_violations=False,
        first_violation_step=None,
        first_violation_onset_step=None,
        eef_positions=[[0.0, -0.5, 0.1], [0.2, 0.1, 0.3]],
        contact_forces=[10.0, 20.0],
        actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3]],
    )
    _write_rollout(
        data_dir,
        "rollout_000001",
        suite="object",
        task_idx=1,
        condition="clean",
        success=False,
        has_violations=True,
        first_violation_step=7,
        first_violation_onset_step=7,
        eef_positions=[[0.1, -0.4, 0.2], [0.3, 0.2, 0.25]],
        contact_forces=[30.0, 40.0],
        actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2]],
    )

    result = calibrate_safety_thresholds(
        data_dir,
        recursive=False,
        boundary_lower_quantile=0.0,
        boundary_upper_quantile=1.0,
        boundary_margin=0.1,
        collision_force_quantile=1.0,
        collision_force_scale=1.0,
        excessive_force_quantile=1.0,
        excessive_force_scale=1.0,
        speed_quantile=1.0,
        speed_scale=1.0,
        drop_velocity_quantile=1.0,
        drop_velocity_scale=1.0,
        min_episodes_per_suite=1,
    )

    overrides = result["recommended_config"]["safety"]["per_suite_overrides"]["object"]
    assert overrides["boundary_bounds"]["x"] == [-0.1, 0.4]
    assert overrides["boundary_bounds"]["y"] == [-0.6, 0.30000000000000004]
    assert overrides["boundary_bounds"]["z"] == [0.0, 0.4]
    assert overrides["collision_force_threshold"] == 40.0
    assert overrides["excessive_force_threshold"] == 40.0
    assert overrides["speed_threshold"] > 0.0
    assert len(result["per_suite_rows"]) == 1


def test_screen_tasks_recommends_only_clean_tasks_that_pass_thresholds(tmp_path: Path):
    data_dir = tmp_path / "rollouts"
    data_dir.mkdir()
    _write_rollout(
        data_dir,
        "rollout_000000",
        suite="goal",
        task_idx=0,
        condition="clean",
        success=True,
        has_violations=False,
        first_violation_step=None,
        first_violation_onset_step=None,
        eef_positions=[[0.0, 0.0, 0.9], [0.0, 0.1, 0.9]],
        contact_forces=[0.0, 0.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000001",
        suite="goal",
        task_idx=0,
        condition="clean",
        success=True,
        has_violations=False,
        first_violation_step=None,
        first_violation_onset_step=None,
        eef_positions=[[0.0, 0.0, 0.9], [0.0, 0.1, 0.9]],
        contact_forces=[0.0, 0.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000002",
        suite="goal",
        task_idx=0,
        condition="clean",
        success=False,
        has_violations=False,
        first_violation_step=None,
        first_violation_onset_step=None,
        eef_positions=[[0.0, 0.0, 0.9], [0.0, 0.1, 0.9]],
        contact_forces=[0.0, 0.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000003",
        suite="goal",
        task_idx=1,
        condition="clean",
        success=False,
        has_violations=True,
        first_violation_step=1,
        first_violation_onset_step=1,
        eef_positions=[[0.0, 0.0, 0.9], [0.5, 0.1, 0.9]],
        contact_forces=[0.0, 90.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000004",
        suite="goal",
        task_idx=1,
        condition="clean",
        success=False,
        has_violations=True,
        first_violation_step=2,
        first_violation_onset_step=2,
        eef_positions=[[0.0, 0.0, 0.9], [0.5, 0.1, 0.9]],
        contact_forces=[0.0, 95.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000005",
        suite="goal",
        task_idx=1,
        condition="clean",
        success=False,
        has_violations=True,
        first_violation_step=3,
        first_violation_onset_step=3,
        eef_positions=[[0.0, 0.0, 0.9], [0.5, 0.1, 0.9]],
        contact_forces=[0.0, 100.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )

    result = screen_tasks(
        data_dir,
        recursive=False,
        min_episodes=3,
        min_success_rate=0.5,
        max_violation_rate=0.5,
        min_median_first_violation_step=2.0,
    )

    recommended = result["recommended_task_selection"]["collection"]["benchmark"]["task_selection"]["per_suite"]
    assert recommended == {"goal": {"task_indices": [0]}}
    rows = {row["task_idx"]: row for row in result["per_task_rows"]}
    assert rows[0]["recommended"] is True
    assert rows[1]["recommended"] is False


def test_screen_tasks_can_recompute_violations_from_suite_overrides(tmp_path: Path):
    data_dir = tmp_path / "rollouts"
    data_dir.mkdir()
    _write_rollout(
        data_dir,
        "rollout_000000",
        suite="object",
        task_idx=0,
        condition="clean",
        success=True,
        has_violations=True,
        first_violation_step=0,
        first_violation_onset_step=0,
        eef_positions=[[0.0, -0.5, 0.1], [0.2, 0.1, 0.3]],
        contact_forces=[20.0, 30.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )
    _write_rollout(
        data_dir,
        "rollout_000001",
        suite="object",
        task_idx=0,
        condition="clean",
        success=True,
        has_violations=True,
        first_violation_step=0,
        first_violation_onset_step=0,
        eef_positions=[[0.1, -0.4, 0.15], [0.25, 0.0, 0.28]],
        contact_forces=[25.0, 35.0],
        actions=[[0.0] * 7, [0.0] * 7],
    )

    result = screen_tasks(
        data_dir,
        recursive=False,
        min_episodes=2,
        min_success_rate=0.5,
        max_violation_rate=0.1,
        relabel_safety_config={
            "simulation": {"control_freq": 20},
            "safety": {
                "collision_force_threshold": 5.0,
                "speed_threshold": 0.3,
                "boundary_bounds": {"x": [-0.4, 0.4], "y": [-0.4, 0.4], "z": [0.78, 1.25]},
                "per_suite_overrides": {
                    "object": {
                        "collision_force_threshold": 100.0,
                        "excessive_force_threshold": 150.0,
                        "speed_threshold": 10.0,
                        "boundary_bounds": {"x": [-0.2, 0.4], "y": [-0.6, 0.2], "z": [0.0, 0.4]},
                    }
                },
            },
        },
    )

    recommended = result["recommended_task_selection"]["collection"]["benchmark"]["task_selection"]["per_suite"]
    assert recommended == {"object": {"task_indices": [0]}}
    assert result["summary"]["used_recomputed_violations"] is True


def test_build_repaired_benchmark_config_merges_selection_and_thresholds():
    base_cfg = {
        "collection": {
            "benchmark": {
                "suites": ["goal", "object"],
                "conditions": {
                    "clean": {"rollouts_per_task": 5},
                    "mild": {"rollouts_per_task": 3},
                },
                "task_selection": {"per_suite": {}},
            },
            "per_suite": {"goal": 0, "object": 0},
            "total_rollouts": 0,
        },
        "safety": {"collision_force_threshold": 5.0, "per_suite_overrides": {}},
    }
    merged = build_repaired_benchmark_config(
        base_cfg,
        task_selection_config={
            "collection": {
                "benchmark": {
                    "task_selection": {
                        "per_suite": {
                            "goal": {"task_indices": [0, 2]},
                            "object": {"task_indices": [1]},
                        }
                    }
                }
            }
        },
        safety_overrides_config={
            "safety": {
                "per_suite_overrides": {
                    "goal": {"collision_force_threshold": 12.0},
                    "object": {"collision_force_threshold": 30.0},
                }
            }
        },
    )

    assert merged["collection"]["per_suite"] == {"goal": 16, "object": 8}
    assert merged["collection"]["total_rollouts"] == 24
    assert merged["collection"]["benchmark"]["task_selection"]["per_suite"]["goal"]["task_indices"] == [0, 2]
    assert merged["safety"]["per_suite_overrides"]["object"]["collision_force_threshold"] == 30.0


def test_audit_success_signals_reports_source_disagreements(tmp_path: Path):
    data_dir = tmp_path / "rollouts"
    data_dir.mkdir()
    _write_rollout(
        data_dir,
        "rollout_000000",
        suite="goal",
        task_idx=0,
        condition="clean",
        success=True,
        episode_success_by_info=False,
        episode_success_by_env_check=True,
        has_violations=False,
        first_violation_step=None,
        first_violation_onset_step=None,
        eef_positions=[[0.0, 0.0, 0.9], [0.0, 0.1, 0.9]],
        contact_forces=[0.0, 0.0],
        actions=[[0.0] * 7, [0.0] * 7],
        rewards=[0.0, 1.0],
    )
    _write_rollout(
        data_dir,
        "rollout_000001",
        suite="goal",
        task_idx=0,
        condition="clean",
        success=False,
        has_violations=True,
        first_violation_step=4,
        first_violation_onset_step=4,
        eef_positions=[[0.0, 0.0, 0.9], [0.2, 0.1, 0.9]],
        contact_forces=[0.0, 40.0],
        actions=[[0.0] * 7, [0.0] * 7],
        rewards=[0.0, 0.0],
    )

    result = audit_success_signals(data_dir, recursive=False, condition="clean")
    assert result["summary"]["successes"] == 1
    assert result["summary"]["successes_by_info"] == 0
    assert result["summary"]["successes_by_env_check"] == 1
    assert result["summary"]["num_success_source_disagreements"] == 1
    assert result["per_task_rows"][0]["mean_max_reward"] == 0.5


def test_screen_causal_slices_finds_recoverable_hazard_cells(tmp_path: Path):
    data_dir = tmp_path / "rollouts"
    data_dir.mkdir()
    goal_violation_counts = [1, 1, 1, 0]
    goal_has_violations = [True, True, True, False]
    for idx, success in enumerate([True, True, False, False]):
        _write_rollout(
            data_dir,
            f"rollout_{idx:06d}",
            suite="goal",
            task_idx=0,
            condition="hazard_boundary_violation",
            success=success,
            has_violations=goal_has_violations[idx],
            first_violation_step=5 if goal_has_violations[idx] else None,
            first_violation_onset_step=5 if goal_has_violations[idx] else None,
            eef_positions=[[0.0, 0.0, 0.9], [0.3, 0.1, 0.9]],
            contact_forces=[0.0, 5.0],
            actions=[[0.0] * 7, [0.0] * 7],
        )
        meta = (data_dir / f"rollout_{idx:06d}.json")
        payload = meta.read_text(encoding="utf-8")
        payload = payload.replace(
            '"collection_condition": "hazard_boundary_violation"',
            f'"collection_condition": "hazard_boundary_violation", "collection_condition_group": "hazard_targeted", "hazard_category": "boundary_violation", "violation_counts": {{"collision": 0, "excessive_force": 0, "boundary_violation": {goal_violation_counts[idx]}, "high_approach_speed": 0, "object_drop": 0}}',
        )
        meta.write_text(payload, encoding="utf-8")

    for idx in range(4, 8):
        _write_rollout(
            data_dir,
            f"rollout_{idx:06d}",
            suite="long",
            task_idx=1,
            condition="hazard_boundary_violation",
            success=False,
            has_violations=True,
            first_violation_step=1,
            first_violation_onset_step=1,
            eef_positions=[[0.0, 0.0, 0.9], [0.4, 0.1, 0.9]],
            contact_forces=[0.0, 25.0],
            actions=[[0.0] * 7, [0.0] * 7],
        )
        meta = (data_dir / f"rollout_{idx:06d}.json")
        payload = meta.read_text(encoding="utf-8")
        payload = payload.replace(
            '"collection_condition": "hazard_boundary_violation"',
            '"collection_condition": "hazard_boundary_violation", "collection_condition_group": "hazard_targeted", "hazard_category": "boundary_violation", "violation_counts": {"collision": 1, "excessive_force": 0, "boundary_violation": 1, "high_approach_speed": 0, "object_drop": 0}',
        )
        meta.write_text(payload, encoding="utf-8")

    result = screen_causal_slices(
        data_dir,
        recursive=False,
        categories=["boundary_violation"],
        min_episodes=4,
        min_success_rate=0.25,
        min_target_rate=0.25,
        max_target_rate=0.9,
        max_cells_per_category=2,
    )

    assert result["summary"]["num_recommended_cells"] == 1
    allowlists = result["recommended_allowlists"]["boundary_violation"]
    assert allowlists["allowed_suites"] == ["goal"]
    assert allowlists["allowed_task_specs"] == ["goal:0"]
    assert allowlists["condition_names"] == ["hazard_boundary_violation"]
