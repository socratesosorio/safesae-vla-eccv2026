import os
import subprocess
from pathlib import Path


def test_cloud_preflight_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.run(
        ["python", "scripts/cloud_preflight.py", "--provider", "both"],
        check=True,
        env=env,
    )
