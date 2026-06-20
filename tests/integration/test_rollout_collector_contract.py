import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION", "0") != "1", reason="Set RUN_INTEGRATION=1 to run")
def test_rollout_collector_contract_imports():
    from src.data.rollout_collector import RolloutCollector

    cfg = {
        "model": {
            "name": "openvla/openvla-7b",
            "checkpoints": {
                "spatial": "openvla/openvla-7b-finetuned-libero-spatial",
                "object": "openvla/openvla-7b-finetuned-libero-object",
                "goal": "openvla/openvla-7b-finetuned-libero-goal",
                "long": "openvla/openvla-7b",
            },
            "dtype": "bfloat16",
        },
        "simulation": {
            "max_episode_length": 10,
            "control_freq": 20,
            "camera_names": ["agentview"],
            "image_size": 224,
        },
        "collection": {
            "total_rollouts": 1,
            "per_suite": {"spatial": 1},
            "noise_fraction": 0.0,
            "noise_std": 0.05,
            "seed": 42,
        },
        "activation_caching": {"layers": [16], "token_positions": "action_only", "dtype": "float16"},
        "safety": {},
        "output": {"base_dir": "outputs/rollouts"},
    }
    collector = RolloutCollector(cfg)
    assert hasattr(collector, "collect_single_rollout")
