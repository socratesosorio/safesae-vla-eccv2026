"""Quick pipeline validation before full rollout spend."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.rollout_collector import RolloutCollector
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight rollout pipeline tests")
    parser.add_argument("--config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--pi0_config", type=str, default="configs/rollout_pi0.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/test_pipeline")
    parser.add_argument("--num_rollouts", type=int, default=50)
    parser.add_argument("--suite", type=str, default="spatial")
    parser.add_argument("--test_pi0", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.output_dir)

    # 1) collector/model construction
    collector = RolloutCollector(cfg)
    assert collector.processor is not None
    assert collector.dtype in {torch.bfloat16, torch.float16}

    # 2) short collection run
    suite = args.suite
    total = int(args.num_rollouts)
    suite_obj = collector._suite_builder(suite)
    if hasattr(suite_obj, "get_num_tasks"):
        n_tasks = int(suite_obj.get_num_tasks())
    elif hasattr(suite_obj, "get_task_names"):
        n_tasks = len(suite_obj.get_task_names())
    elif hasattr(suite_obj, "n_tasks"):
        n_tasks = int(suite_obj.n_tasks)
    else:
        n_tasks = max(total, 1)
    schedule = []
    for i in range(total):
        noise_std = 0.0 if i < total // 2 else float(cfg.get("collection", {}).get("noise", {}).get("mild_noise_std", 0.03))
        schedule.append((suite, i % max(n_tasks, 1), noise_std))

    for i, (suite_name, task_idx, noise_level) in enumerate(schedule):
        rollout = collector.collect_single_rollout(task_idx=task_idx, suite=suite_name, add_noise=noise_level)
        rid = f"rollout_{i:05d}"
        collector._save_rollout(rollout, out_dir, rid)

    files = sorted(glob.glob(str(out_dir / "rollout_*.safetensors")))
    assert len(files) == total, f"Expected {total} rollout files, found {len(files)}"

    # 3) tensor contract check
    sample = load_file(files[0])
    assert "actions" in sample
    assert "safety_labels" in sample
    layer_keys = sorted([k for k in sample.keys() if k.startswith("activations_layer")])
    expected_layers = [int(x) for x in cfg.get("activation_caching", {}).get("layers", [20])]
    assert len(layer_keys) == len(expected_layers), f"Expected {len(expected_layers)} layer keys, got {layer_keys}"
    for layer in expected_layers:
        key = f"activations_layer{layer}"
        assert key in sample, f"Missing tensor key {key}"
        acts = sample[key]
        assert acts.shape[-1] == 4096
        assert acts.shape[1] == 7
        assert not torch.isnan(acts.to(torch.float32)).any()

    # 4) violation distribution sanity
    has_viol = []
    for f in files:
        t = load_file(f)
        has_viol.append(bool(t["safety_labels"].any().item()))
    violation_rate = float(np.mean(has_viol)) if has_viol else 0.0

    report = {
        "num_rollouts": total,
        "violation_rate": violation_rate,
        "warnings": [],
    }
    if violation_rate < 0.10:
        report["warnings"].append("Violation rate <10%; consider increasing noise.")
    if violation_rate > 0.90:
        report["warnings"].append("Violation rate >90%; consider reducing noise.")

    with (out_dir / "pipeline_test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.test_pi0:
        from src.data.pi0_rollout_collector import Pi0RolloutCollector

        pi0_cfg = load_yaml(args.pi0_config)
        pi0_out = ensure_dir(out_dir / "pi0")
        pi0_collector = Pi0RolloutCollector(pi0_cfg)
        rollout = pi0_collector.collect_single_rollout(task_idx=0, suite="spatial", noise_level=0.0)
        pi0_collector._save_rollout(rollout, pi0_out, "rollout_000000")
        pi0_sample = load_file(str(pi0_out / "rollout_000000.safetensors"))
        for layer in [int(x) for x in pi0_cfg.get("activation_caching", {}).get("layers", [9, 11, 14])]:
            key = f"activations_layer{layer}"
            assert key in pi0_sample, f"Missing pi0 activation key {key}"
            acts = pi0_sample[key]
            assert acts.shape[-1] == int(pi0_cfg.get("activation_caching", {}).get("d_in", 2048))
            assert not torch.isnan(acts.to(torch.float32)).any()

    print(json.dumps(report, indent=2))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
