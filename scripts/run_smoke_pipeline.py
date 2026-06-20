"""Generate synthetic rollout artifacts and run a fast end-to-end smoke pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic smoke pipeline")
    parser.add_argument("--num_episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output_root", type=str, default="outputs/smoke")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_synthetic_rollouts(data_dir: Path, n_eps: int, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(n_eps):
        unsafe = ep % 3 == 0
        acts16 = rng.normal(0, 0.4, size=(steps, 7, 4096)).astype(np.float32)
        acts24 = rng.normal(0, 0.4, size=(steps, 7, 4096)).astype(np.float32)

        # Inject a synthetic safety signal into a small feature subspace for unsafe episodes.
        if unsafe:
            acts16[:, :, 128:136] += 2.0
            acts24[:, :, 512:520] += 2.5

        safety_labels = np.zeros((steps, 5), dtype=np.bool_)
        if unsafe:
            idx = rng.choice(np.arange(steps), size=max(2, steps // 5), replace=False)
            safety_labels[idx, 0] = True
            safety_labels[idx[: max(1, len(idx) // 2)], 1] = True

        episode_viol = safety_labels.sum(axis=0).astype(np.int32)
        tensors = {
            "activations_layer16": torch.from_numpy(acts16.astype(np.float16)),
            "activations_layer24": torch.from_numpy(acts24.astype(np.float16)),
            "actions": torch.from_numpy(rng.normal(0, 0.2, size=(steps, 7)).astype(np.float32)),
            "eef_positions": torch.from_numpy(rng.normal(0, 0.1, size=(steps, 3)).astype(np.float32)),
            "contact_forces": torch.from_numpy(rng.uniform(0, 60, size=(steps,)).astype(np.float32)),
            "safety_labels": torch.from_numpy(safety_labels),
            "episode_success": torch.tensor([not unsafe], dtype=torch.bool),
            "episode_safety_violations": torch.from_numpy(episode_viol),
        }

        rid = f"rollout_{ep:06d}"
        save_file(tensors, str(data_dir / f"{rid}.safetensors"))
        with (data_dir / f"{rid}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "suite": "smoke",
                    "task_idx": ep,
                    "instruction": "synthetic task",
                    "checkpoint": "synthetic",
                    "noise_applied": False,
                    "num_steps": steps,
                    "timestamp": 0,
                    "safety_categories": [
                        "collision",
                        "excessive_force",
                        "boundary_violation",
                        "high_approach_speed",
                        "object_drop",
                    ],
                    "action_dim": 7,
                },
                f,
                indent=2,
            )


def write_smoke_sae_config(base_path: Path) -> Path:
    cfg = {
        "sae": {
            "architecture": "batch_topk",
            "d_in": 4096,
            "d_sae": 2048,
            "k": 32,
        },
        "training": {
            "lr": 5e-4,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 100,
            "lr_decay_start": 0.8,
            "batch_size": 512,
            "total_training_tokens": 50000,
            "normalize_activations": "expected_average_only_in",
            "checkpoint_interval": 200,
            "eval_interval": 100,
            "max_grad_norm": 1.0,
            "num_workers": 0,
            "seed": 42,
            "wandb_project": "safesae-vla-smoke",
            "wandb_entity": None,
            "log_to_wandb": False,
        },
        "evaluation": {
            "target_l0": [16, 64],
            "max_dead_features_pct": 25.0,
            "target_fvu": 0.5,
        },
    }
    out = base_path / "smoke_sae_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    data_dir = root / "rollouts"
    checkpoints = root / "checkpoints"
    results = root / "results"
    figures = root / "figures"

    write_synthetic_rollouts(data_dir, args.num_episodes, args.steps, args.seed)
    sae_cfg = write_smoke_sae_config(root)

    subprocess.run(
        [
            "python",
            "scripts/run_sae_training.py",
            "--provider",
            "local",
            "--backend",
            "manual",
            "--layer",
            "16",
            "--config",
            str(sae_cfg),
            "--data_dir",
            str(data_dir),
            "--checkpoint_dir",
            str(checkpoints),
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "scripts/run_analysis.py",
            "--sae_checkpoint",
            str(checkpoints / "sae_layer16_final.pt"),
            "--layer",
            "16",
            "--data_dir",
            str(data_dir),
            "--sae_config",
            str(sae_cfg),
            "--output_dir",
            str(results),
            "--generate_figures",
            "--figures_dir",
            str(figures),
            "--paper_dir",
            str(root / "paper"),
        ],
        check=True,
    )

    print(f"Smoke pipeline complete. Results: {results}")


if __name__ == "__main__":
    main()
