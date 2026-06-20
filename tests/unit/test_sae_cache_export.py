from pathlib import Path

import torch
from safetensors.torch import save_file

from src.sae.train_sae import export_saelens_cache
from src.utils.runtime import save_json


def _write_rollout(path: Path, steps: int = 3, unsafe: bool = False):
    episode_violations = (
        torch.tensor([1, 0, 0, 0, 0], dtype=torch.int32) if unsafe else torch.zeros(5, dtype=torch.int32)
    )
    safety_labels = torch.zeros(steps, 5, dtype=torch.bool)
    if unsafe:
        safety_labels[:, 0] = True

    data = {
        "activations_layer16": torch.randn(steps, 7, 4096, dtype=torch.float16),
        "safety_labels": safety_labels,
        "episode_safety_violations": episode_violations,
        "actions": torch.randn(steps, 7),
        "eef_positions": torch.randn(steps, 3),
        "contact_forces": torch.randn(steps),
        "episode_success": torch.tensor([True], dtype=torch.bool),
    }
    save_file(data, str(path))
    save_json(path.with_suffix(".json"), {"num_steps": steps})


def test_export_saelens_cache_creates_manifest(tmp_path: Path):
    for i in range(2):
        _write_rollout(tmp_path / f"rollout_{i:06d}.safetensors", steps=2)

    manifest = export_saelens_cache(
        data_dir=str(tmp_path),
        layer=16,
        tr_cfg={"batch_size": 4, "num_workers": 0},
        cache_dir=str(tmp_path / "cache"),
        test_split=0.0,
        seed=0,
        shard_size=8,
        rebuild=True,
    )
    assert manifest["num_shards"] >= 1
    assert manifest["num_vectors"] > 0


def test_export_saelens_cache_respects_filter_mode_and_rebuilds_on_mismatch(tmp_path: Path):
    _write_rollout(tmp_path / "rollout_safe.safetensors", steps=2, unsafe=False)  # 14 vectors
    _write_rollout(tmp_path / "rollout_unsafe.safetensors", steps=3, unsafe=True)  # 21 vectors

    cache_dir = tmp_path / "cache"
    manifest_safe = export_saelens_cache(
        data_dir=str(tmp_path),
        layer=16,
        tr_cfg={"batch_size": 4, "num_workers": 0},
        cache_dir=str(cache_dir),
        test_split=0.0,
        seed=0,
        shard_size=8,
        rebuild=True,
        filter_mode="safe",
    )
    assert manifest_safe["filter_mode"] == "safe"
    assert manifest_safe["num_vectors"] == 14

    # Reuse same cache directory without rebuild; function should detect filter mismatch and rebuild.
    manifest_unsafe = export_saelens_cache(
        data_dir=str(tmp_path),
        layer=16,
        tr_cfg={"batch_size": 4, "num_workers": 0},
        cache_dir=str(cache_dir),
        test_split=0.0,
        seed=0,
        shard_size=8,
        rebuild=False,
        filter_mode="unsafe",
    )
    assert manifest_unsafe["filter_mode"] == "unsafe"
    assert manifest_unsafe["num_vectors"] == 21


def test_export_saelens_cache_rebuilds_when_source_data_dir_changes(tmp_path: Path):
    data_a = tmp_path / "data_a"
    data_b = tmp_path / "data_b"
    data_a.mkdir(parents=True, exist_ok=True)
    data_b.mkdir(parents=True, exist_ok=True)

    _write_rollout(data_a / "rollout_000000.safetensors", steps=1, unsafe=False)  # 7 vectors
    _write_rollout(data_b / "rollout_000000.safetensors", steps=3, unsafe=False)  # 21 vectors

    cache_dir = tmp_path / "shared_cache"
    manifest_a = export_saelens_cache(
        data_dir=str(data_a),
        layer=16,
        tr_cfg={"batch_size": 4, "num_workers": 0},
        cache_dir=str(cache_dir),
        test_split=0.0,
        seed=0,
        shard_size=8,
        rebuild=True,
    )
    assert manifest_a["num_vectors"] == 7

    # Reusing same cache directory with a different rollout source should force rebuild.
    manifest_b = export_saelens_cache(
        data_dir=str(data_b),
        layer=16,
        tr_cfg={"batch_size": 4, "num_workers": 0},
        cache_dir=str(cache_dir),
        test_split=0.0,
        seed=0,
        shard_size=8,
        rebuild=False,
    )
    assert manifest_b["num_vectors"] == 21
    assert manifest_a["source_data_dir"] != manifest_b["source_data_dir"]
    assert manifest_a["source_fingerprint"] != manifest_b["source_fingerprint"]
