import torch

from src.sae.train_sae import validate_cached_activation_store


class _TinyActivationDataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        return {
            "activation": torch.randn(4096, dtype=torch.float32),
            "safety_label": torch.zeros(5, dtype=torch.bool),
            "episode_path": "dummy.safetensors",
            "episode_idx": 0,
            "step_idx": idx,
            "token_idx": 0,
        }


class _EmptyBatchStore:
    """Mimics CachedActivationsStore behavior when drop_last=True yields zero batches."""

    def __init__(self, *args, **kwargs):
        self.dataset = _TinyActivationDataset()

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        return iter(())


def test_validate_cached_activation_store_handles_empty_train_batches(monkeypatch):
    monkeypatch.setattr("src.sae.train_sae.CachedActivationsStore", _EmptyBatchStore)

    info = validate_cached_activation_store(
        data_dir="unused",
        layer=16,
        tr_cfg={"batch_size": 8, "num_workers": 0, "seed": 0},
        test_split=0.2,
        filter_mode="unsafe",
    )

    assert info["num_vectors"] == 3
    assert info["batch_size"] == 8
    assert info["effective_train_batches"] == 0
    assert info["first_batch_shape"] == [3, 4096]
