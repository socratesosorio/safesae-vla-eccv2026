from pathlib import Path
import json

import torch
from safetensors.torch import save_file

from src.data.activation_dataset import AnalysisDataset
from src.data.data_utils import build_temporal_event_metadata, train_test_split_paths
from src.utils.runtime import save_json


def _write_rollout(path: Path, task_idx: int, suite: str, steps: int = 3, metadata_extra: dict | None = None):
    payload = {
        "activations_layer16": torch.randn(steps, 7, 4096, dtype=torch.float16),
        "safety_labels": torch.zeros(steps, 5, dtype=torch.bool),
        "episode_safety_violations": torch.zeros(5, dtype=torch.int32),
        "actions": torch.randn(steps, 7),
        "eef_positions": torch.randn(steps, 3),
        "contact_forces": torch.randn(steps),
        "episode_success": torch.tensor([False], dtype=torch.bool),
    }
    save_file(payload, str(path))
    metadata = {
        "suite": suite,
        "task_idx": task_idx,
        "num_steps": steps,
        "episode_success": False,
        "episode_failure": True,
        "violation_counts": {
            "collision": 0,
            "excessive_force": 0,
            "boundary_violation": 0,
            "high_approach_speed": 0,
            "object_drop": 0,
        },
        "total_violations": 0,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    save_json(path.with_suffix(".json"), metadata)


def test_train_test_split_paths_task_mode_keeps_tasks_together(tmp_path: Path):
    paths = []
    for idx, task_idx in enumerate([0, 0, 1, 1]):
        path = tmp_path / f"rollout_{idx:06d}.safetensors"
        _write_rollout(path, task_idx=task_idx, suite="goal")
        paths.append(path)

    train, test = train_test_split_paths(paths, test_split=0.5, seed=0, split_mode="task")
    assert len(train) + len(test) == len(paths)

    def _task_ids(group):
        ids = set()
        for path in group:
            with Path(path).with_suffix(".json").open("r", encoding="utf-8") as f:
                meta = json.load(f)
            ids.add((meta["suite"], int(meta["task_idx"])))
        return ids

    train_tasks = _task_ids(train)
    test_tasks = _task_ids(test)
    assert train_tasks
    assert test_tasks
    assert train_tasks.isdisjoint(test_tasks)


def test_build_temporal_event_metadata_computes_first_steps():
    safety_matrix = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    ).numpy()
    result = build_temporal_event_metadata(
        success_flags=[False, False, True, True],
        safety_matrix=safety_matrix,
        categories=[
            "collision",
            "excessive_force",
            "boundary_violation",
            "high_approach_speed",
            "object_drop",
        ],
    )

    assert result["success_by_timestep"] == [False, False, True, True]
    assert result["first_success_step"] == 2
    assert result["violation_by_timestep"] == [False, True, False, True]
    assert result["first_violation_step"] == 1
    assert result["first_violation_step_by_category"]["collision"] == 1
    assert result["first_violation_step_by_category"]["excessive_force"] == 3
    assert result["violation_by_timestep_by_category"]["collision"] == [False, True, False, False]
    assert result["first_violation_step_by_category"]["object_drop"] is None


def test_build_temporal_event_metadata_computes_onsets_with_dwell():
    safety_matrix = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    ).numpy()
    result = build_temporal_event_metadata(
        success_flags=[False] * 6,
        safety_matrix=safety_matrix,
        categories=[
            "collision",
            "excessive_force",
            "boundary_violation",
            "high_approach_speed",
            "object_drop",
        ],
        min_active_steps_by_category={"collision": 2, "excessive_force": 2},
    )

    assert result["violation_onset_by_timestep"] == [True, False, True, True, False, False]
    assert result["first_violation_onset_step"] == 0
    assert result["violation_onset_by_timestep_by_category"]["collision"] == [True, False, False, True, False, False]
    assert result["violation_onset_by_timestep_by_category"]["excessive_force"] == [False, False, True, False, False, False]
    assert result["violation_onset_counts_by_category"]["collision"] == 2
    assert result["violation_onset_counts_by_category"]["excessive_force"] == 1
    assert result["first_violation_onset_step_by_category"]["collision"] == 0
    assert result["first_violation_onset_step_by_category"]["excessive_force"] == 2


def test_analysis_dataset_task_split_groups_tasks(tmp_path: Path):
    for idx, task_idx in enumerate([0, 0, 1, 1]):
        _write_rollout(tmp_path / f"rollout_{idx:06d}.safetensors", task_idx=task_idx, suite="goal")

    dataset = AnalysisDataset(data_dir=str(tmp_path), test_split=0.5, seed=0, split_mode="task")

    train_groups = {dataset.group_ids[idx] for idx in dataset.train_indices}
    test_groups = {dataset.group_ids[idx] for idx in dataset.test_indices}
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)


def test_analysis_dataset_prefix_helpers(tmp_path: Path):
    _write_rollout(
        tmp_path / "rollout_000000.safetensors",
        task_idx=0,
        suite="goal",
        steps=3,
        metadata_extra={
            "success_by_timestep": [False, False, False],
            "violation_by_timestep": [False, True, False],
            "violation_onset_by_timestep": [False, True, False],
            "violation_by_timestep_by_category": {
                "collision": [False, True, False],
                "excessive_force": [False, False, False],
                "boundary_violation": [False, False, False],
                "high_approach_speed": [False, False, False],
                "object_drop": [False, False, False],
            },
            "violation_onset_by_timestep_by_category": {
                "collision": [False, True, False],
                "excessive_force": [False, False, False],
                "boundary_violation": [False, False, False],
                "high_approach_speed": [False, False, False],
                "object_drop": [False, False, False],
            },
            "first_violation_step": 1,
            "first_violation_onset_step": 1,
            "first_violation_step_by_category": {
                "collision": 1,
                "excessive_force": None,
                "boundary_violation": None,
                "high_approach_speed": None,
                "object_drop": None,
            },
            "first_violation_onset_step_by_category": {
                "collision": 1,
                "excessive_force": None,
                "boundary_violation": None,
                "high_approach_speed": None,
                "object_drop": None,
            },
            "violation_onset_counts_by_category": {
                "collision": 1,
                "excessive_force": 0,
                "boundary_violation": 0,
                "high_approach_speed": 0,
                "object_drop": 0,
            },
            "total_violation_onsets": 1,
        },
    )

    dataset = AnalysisDataset(data_dir=str(tmp_path), test_split=0.5, seed=0, split_mode="episode")
    prefix = dataset.get_prefix_item(0, end_step=2)
    assert prefix["actions"].shape[0] == 2
    assert prefix["activations_layer16"].shape[0] == 2
    assert prefix["metadata"]["prefix_end_step"] == 2
    assert dataset.get_future_window_label(0, prefix_end_step=1, horizon=2, target="violation") == 1
    assert dataset.get_future_window_label(0, prefix_end_step=2, horizon=1, target="violation") == 0
    assert dataset.get_future_window_label(
        0,
        prefix_end_step=1,
        horizon=2,
        target="category_violation",
        category="collision",
    ) == 1
    assert dataset.get_future_window_label(0, prefix_end_step=1, horizon=2, target="violation_onset") == 1
    assert dataset.get_future_window_label(
        0,
        prefix_end_step=1,
        horizon=2,
        target="category_violation_onset",
        category="collision",
    ) == 1
    assert dataset.get_future_window_label(0, prefix_end_step=1, horizon=2, target="episode_failure") == 1
