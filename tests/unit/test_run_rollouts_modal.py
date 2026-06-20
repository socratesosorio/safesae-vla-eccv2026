import importlib.util
from pathlib import Path

import pytest


def _load_module(path: Path):
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("run_rollouts_modal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_worker_count_prefers_config_when_cli_omitted():
    module = _load_module(Path("scripts/run_rollouts_modal.py"))
    assert module._resolve_worker_count(None, {"num_workers": 5}) == 5
    assert module._resolve_worker_count(2, {"num_workers": 5}) == 2


def test_build_collect_rollouts_fn_binds_selected_modal_config(monkeypatch):
    module = _load_module(Path("scripts/run_rollouts_modal.py"))
    captured = {}

    class _DummyVolume:
        pass

    def fake_from_name(name, create_if_missing=False):
        captured["volume_name"] = name
        captured["create_if_missing"] = create_if_missing
        return _DummyVolume()

    def fake_app_function(**kwargs):
        captured["app_kwargs"] = kwargs

        def decorator(fn):
            captured["wrapped_fn"] = fn
            return fn

        return decorator

    monkeypatch.setattr(module.modal.Volume, "from_name", staticmethod(fake_from_name))
    monkeypatch.setattr(module.app, "function", fake_app_function)
    monkeypatch.setattr(module, "_gpu_spec", lambda cfg: "GPU_SENTINEL")

    cfg = {"mount_path": "/mnt/custom", "volume_name": "custom-volume", "timeout_sec": 321}
    fn = module._build_collect_rollouts_fn(cfg)

    assert fn is module._collect_rollouts_chunk_impl
    assert captured["volume_name"] == "custom-volume"
    assert captured["create_if_missing"] is True
    assert list(captured["app_kwargs"]["volumes"].keys()) == ["/mnt/custom"]
    assert captured["app_kwargs"]["timeout"] == 321
    assert captured["app_kwargs"]["gpu"] == "GPU_SENTINEL"


def test_rollout_modal_config_loaders_expand_env_vars(tmp_path: Path, monkeypatch):
    module = _load_module(Path("scripts/run_rollouts_modal.py"))

    monkeypatch.setenv("MODAL_TEST_VOLUME", "env-volume")
    monkeypatch.setenv("ROLLOUT_TEST_OUTPUT", "/tmp/env-rollouts")

    modal_cfg = tmp_path / "modal.yaml"
    modal_cfg.write_text("modal:\n  volume_name: ${MODAL_TEST_VOLUME}\n", encoding="utf-8")
    loaded_modal = module.load_modal_cfg(str(modal_cfg))
    assert loaded_modal["volume_name"] == "env-volume"

    rollout_cfg = tmp_path / "rollout.yaml"
    rollout_cfg.write_text("output:\n  base_dir: ${ROLLOUT_TEST_OUTPUT}\n", encoding="utf-8")
    loaded_rollout = module.load_rollout_config(str(rollout_cfg))
    assert loaded_rollout["output"]["base_dir"] == "/tmp/env-rollouts"
