from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.utils.runtime import save_json

pytest.importorskip("sae_lens")

from src.data.activation_dataset import ActivationDataset, FlattenedActivationDataset


def _write_rollout(path: Path, steps: int = 3):
    data = {
        "activations_layer16": torch.randn(steps, 7, 4096, dtype=torch.float16),
        "safety_labels": torch.zeros(steps, 5, dtype=torch.bool),
        "episode_safety_violations": torch.zeros(5, dtype=torch.int32),
        "actions": torch.randn(steps, 7),
        "eef_positions": torch.randn(steps, 3),
        "contact_forces": torch.randn(steps),
        "episode_success": torch.tensor([True], dtype=torch.bool),
    }
    save_file(data, str(path))
    save_json(path.with_suffix(".json"), {"num_steps": steps})


def test_activation_dataset_episode_read(tmp_path: Path):
    f = tmp_path / "rollout_000000.safetensors"
    _write_rollout(f, steps=4)

    ds = ActivationDataset(data_dir=str(tmp_path), layer=16, split="all")
    item = ds[0]
    assert item["activations"].shape == (4, 7, 4096)
    assert item["safety_labels"].shape == (4, 7, 5)


def test_flattened_dataset_indexing(tmp_path: Path):
    f = tmp_path / "rollout_000000.safetensors"
    _write_rollout(f, steps=2)

    ds = FlattenedActivationDataset(data_dir=str(tmp_path), layer=16, split="all")
    assert len(ds) == 14
    item = ds[5]
    assert item["activation"].shape[0] == 4096


def _write_pi0_rollout(path: Path, steps: int = 3):
    """Write a pi0-style rollout with [T, 1, 2048] activations."""
    data = {
        "activations_layer11": torch.randn(steps, 1, 2048, dtype=torch.float16),
        "safety_labels": torch.zeros(steps, 5, dtype=torch.bool),
        "episode_safety_violations": torch.zeros(5, dtype=torch.int32),
        "actions": torch.randn(steps, 7),
        "eef_positions": torch.randn(steps, 3),
        "contact_forces": torch.randn(steps),
        "episode_success": torch.tensor([True], dtype=torch.bool),
    }
    save_file(data, str(path))
    save_json(path.with_suffix(".json"), {"num_steps": steps})


def test_activation_dataset_pi0_shape(tmp_path: Path):
    f = tmp_path / "rollout_000000.safetensors"
    _write_pi0_rollout(f, steps=4)

    ds = ActivationDataset(data_dir=str(tmp_path), layer=11, split="all")
    item = ds[0]
    assert item["activations"].shape == (4, 1, 2048)
    assert item["safety_labels"].shape == (4, 1, 5)


def test_flattened_dataset_pi0_indexing(tmp_path: Path):
    f = tmp_path / "rollout_000000.safetensors"
    _write_pi0_rollout(f, steps=5)

    ds = FlattenedActivationDataset(data_dir=str(tmp_path), layer=11, split="all")
    # pi0: tokens_per_step=1, so len == num_steps * 1
    assert len(ds) == 5
    item = ds[3]
    assert item["activation"].shape[0] == 2048
    assert item["step_idx"] == 3
    assert item["token_idx"] == 0
