"""Direct PyTorch training loop for BatchTopK SAE (no SAELens dependency)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.data.activation_dataset import SAETrainingDataset
from src.sae.model import BatchTopKSAE
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir, set_global_seed

LOGGER = logging.getLogger(__name__)


def batch_iterator(dataset: SAETrainingDataset, batch_size: int):
    batch = []
    for x in dataset:
        batch.append(x)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch = []
    if batch:
        yield torch.stack(batch, dim=0)


def estimate_norm_factor(data_dir: str, layer: int, sample_limit: int = 10_000, seed: int = 42) -> float:
    ds = SAETrainingDataset(data_dir=data_dir, layer=layer, shuffle=True, seed=seed)
    collected = []
    for i, x in enumerate(ds):
        collected.append(x)
        if i + 1 >= sample_limit:
            break
    if not collected:
        return 1.0
    stacked = torch.stack(collected, dim=0)
    return float(stacked.norm(dim=-1).mean().item())


def _checkpoint_path(output_dir: Path, prefix: str, step: int) -> Path:
    return output_dir / f"{prefix}_step_{step:06d}.pt"


def _latest_checkpoint(output_dir: Path, prefix: str) -> Path | None:
    ckpts = sorted(output_dir.glob(f"{prefix}_step_*.pt"))
    return ckpts[-1] if ckpts else None


def train_sae(
    config_path: str,
    data_dir: str,
    output_dir: str,
    layer: int = 20,
    d_sae: int | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    tr_cfg = cfg.get("training", {})
    set_global_seed(int(tr_cfg.get("seed", 42)))

    output_path = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary = cfg.get("primary", cfg)
    d_in = int(primary.get("d_in", 4096))
    d_sae_final = int(d_sae if d_sae is not None else primary.get("d_sae", 16384))
    k_final = int(k if k is not None else primary.get("k", 32))
    run_prefix = f"sae_layer{int(layer)}_d{d_sae_final}"

    model = BatchTopKSAE(
        d_in=d_in,
        d_sae=d_sae_final,
        k=k_final,
    ).to(device)

    lr = float(tr_cfg.get("lr", 5e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr_cfg.get("weight_decay", 0.0)))
    batch_size = int(tr_cfg.get("batch_size", 4096))
    total_tokens = int(tr_cfg.get("total_training_tokens", 50_000_000))
    total_steps = max(total_tokens // max(batch_size, 1), 1)
    warmup_steps = int(tr_cfg.get("lr_warmup_steps", 500))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    max_grad_norm = float(tr_cfg.get("max_grad_norm", 1.0))

    checkpoint_every = int(tr_cfg.get("checkpoint_every_steps", tr_cfg.get("checkpoint_interval", 2000)))
    eval_every = int(tr_cfg.get("eval_every_steps", tr_cfg.get("eval_interval", 1000)))
    log_every = int(tr_cfg.get("log_every_steps", 100))

    seed_base = int(tr_cfg.get("seed", 42))
    dataset_probe = SAETrainingDataset(
        data_dir=data_dir,
        layer=layer,
        shuffle=False,
        seed=seed_base,
    )
    if not dataset_probe.files:
        raise FileNotFoundError(
            f"No rollout activation files found in {data_dir}. "
            "Expected files matching rollout_*.safetensors."
        )

    norm_factor = estimate_norm_factor(data_dir=data_dir, layer=layer, seed=seed_base)
    norm_factor = max(norm_factor, 1e-6)
    metrics_history: list[dict[str, float]] = []
    start_step = 0

    latest = _latest_checkpoint(output_path, run_prefix)
    if latest is not None:
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        norm_factor = float(ckpt.get("norm_factor", norm_factor))
        metrics_history = ckpt.get("metrics_history", [])
        LOGGER.info("Resuming from %s at step %d", latest, start_step)

    global_step = start_step
    while global_step < total_steps:
        ds = SAETrainingDataset(
            data_dir=data_dir,
            layer=layer,
            shuffle=True,
            seed=seed_base + global_step,
        )
        advanced = False
        for batch in batch_iterator(ds, batch_size=batch_size):
            if global_step >= total_steps:
                break
            model.train()
            x = (batch.to(device=device, dtype=torch.float32) / norm_factor)

            if global_step < warmup_steps:
                scale = float(global_step + 1) / float(max(warmup_steps, 1))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * scale

            loss, metrics = model.compute_loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            model.normalize_decoder()
            if global_step >= warmup_steps:
                scheduler.step()

            global_step += 1

            if global_step % log_every == 0:
                metrics_row = {"step": float(global_step), **metrics}
                metrics_history.append(metrics_row)
                LOGGER.info(
                    "step=%d loss=%.6f l0=%.2f fvu=%.4f dead=%.2f%%",
                    global_step,
                    metrics["loss"],
                    metrics["l0"],
                    metrics["fvu"],
                    metrics["dead_features_pct"],
                )

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    x_hat, acts = model(x[: min(512, x.shape[0])])
                    eval_loss = float((x[: x_hat.shape[0]] - x_hat).pow(2).mean().item())
                    eval_l0 = float((acts > 0).float().sum(dim=-1).mean().item())
                metrics_history.append({"step": float(global_step), "eval_loss": eval_loss, "eval_l0": eval_l0})

            if global_step % checkpoint_every == 0:
                ckpt_state = {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                    "norm_factor": norm_factor,
                    "metrics_history": metrics_history,
                    "layer": int(layer),
                }
                torch.save(ckpt_state, _checkpoint_path(output_path, run_prefix, global_step))
            advanced = True
        if not advanced:
            raise RuntimeError(
                f"No activation vectors were yielded from {data_dir} for layer {layer}. "
                "Check rollout artifacts for missing/empty activation tensors."
            )

    final_ckpt = output_path / f"{run_prefix}.pt"
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "norm_factor": norm_factor,
            "metrics_history": metrics_history,
            "layer": int(layer),
            "d_sae": int(d_sae_final),
            "k": int(k_final),
            "d_in": int(d_in),
        },
        final_ckpt,
    )

    with (output_path / f"{run_prefix}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    return {
        "checkpoint": str(final_ckpt),
        "steps": global_step,
        "norm_factor": norm_factor,
        "output_dir": str(output_path),
        "layer": int(layer),
        "d_sae": int(d_sae_final),
        "k": int(k_final),
        "d_in": int(d_in),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BatchTopK SAE on cached rollout activations")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sae")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--d_sae", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    result = train_sae(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layer=args.layer,
        d_sae=args.d_sae,
        k=args.k,
    )
    LOGGER.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
