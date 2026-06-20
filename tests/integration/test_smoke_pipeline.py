import os
import subprocess

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION", "0") != "1", reason="Set RUN_INTEGRATION=1 to run")
def test_smoke_pipeline_end_to_end():
    subprocess.run(
        ["python", "scripts/run_smoke_pipeline.py", "--num_episodes", "8", "--steps", "20"],
        check=True,
    )
