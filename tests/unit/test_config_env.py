import os
from pathlib import Path

from src.utils.config import load_yaml


def test_load_yaml_env_interpolation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CFG_TEST_PATH", "/tmp/example")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("value: ${CFG_TEST_PATH}/artifact\n", encoding="utf-8")

    cfg = load_yaml(cfg_path)
    assert cfg["value"] == "/tmp/example/artifact"
