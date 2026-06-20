import argparse
import subprocess
from pathlib import Path

import scripts.run_smoke_pipeline as smoke


def test_smoke_pipeline_routes_paper_dir_under_output_root(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "smoke_out"
    calls = []

    monkeypatch.setattr(
        smoke,
        "parse_args",
        lambda: argparse.Namespace(
            num_episodes=2,
            steps=3,
            output_root=str(output_root),
            seed=123,
        ),
    )
    monkeypatch.setattr(smoke, "write_synthetic_rollouts", lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke, "write_smoke_sae_config", lambda base_path: base_path / "smoke_sae_config.yaml")

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    smoke.main()

    assert len(calls) == 2
    analysis_cmd = calls[1]
    assert "--paper_dir" in analysis_cmd
    paper_idx = analysis_cmd.index("--paper_dir") + 1
    assert analysis_cmd[paper_idx] == str(output_root / "paper")
