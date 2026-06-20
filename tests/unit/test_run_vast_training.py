import argparse
import importlib.util
import subprocess
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_vast_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_vast_training_does_not_execute_install_cmd(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "vast_config.yaml"
    cfg.write_text(
        "vast:\n"
        "  install_cmd: \"echo install\"\n"
        "  data_dir: \"/tmp/data/rollouts\"\n"
        "  checkpoint_dir: \"/tmp/checkpoints\"\n",
        encoding="utf-8",
    )

    module = _load_module(Path("scripts/run_vast_training.py"))
    calls = []

    def fake_run(cmd, check=True, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            vast_config=str(cfg),
            layer=16,
            backend="manual",
            sae_config="configs/sae_config.yaml",
        ),
    )

    module.main()

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["python", "scripts/run_sae_training.py", "--provider"]
    assert "vast" in cmd
    assert "echo install" not in " ".join(cmd)
