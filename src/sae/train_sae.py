"""SAE training entrypoint with SAELens and manual BatchTopK backends."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.activation_dataset import CachedActivationsStore, FlattenedActivationDataset
from src.sae.sae_utils import (
    compute_dead_features,
    compute_fvu,
    compute_l0,
    load_checkpoint,
    normalize_expected_average_only_in,
    save_checkpoint,
)
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir, set_global_seed

LOGGER = logging.getLogger(__name__)


def resolve_sae_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Support both legacy `sae` and refactored `primary` config schemas."""
    if "sae" in cfg and isinstance(cfg["sae"], dict):
        return cfg["sae"]
    if "primary" in cfg and isinstance(cfg["primary"], dict):
        return cfg["primary"]
    if isinstance(cfg, dict):
        return cfg
    raise ValueError("Invalid SAE config structure")


class BatchTopKSAE(nn.Module):
    """Manual BatchTopK SAE fallback used when SAELens custom data path is unavailable."""

    def __init__(self, d_in: int = 4096, d_sae: int = 32768, k: int = 48):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=True)
        self.k = int(k)

        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre, k=min(self.k, pre.shape[-1]), dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    # Start cosine restarts around quarter horizon.
    t0 = max(total_steps // 4, 1)
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0)


def maybe_init_wandb(cfg: dict, run_name: str) -> bool:
    if not cfg.get("log_to_wandb", True):
        return False
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime specific.
        LOGGER.warning("W&B unavailable (%s). Continuing without logging.", exc)
        return False
    wandb.init(
        project=cfg.get("wandb_project", "safesae-vla"),
        entity=cfg.get("wandb_entity", None),
        name=run_name,
        config=cfg,
    )
    return True


def build_source_data_identity(data_dir: str) -> dict[str, Any]:
    """Build a lightweight fingerprint for rollout activation source compatibility checks."""
    base = Path(data_dir).expanduser()
    resolved = str(base.resolve())

    files = sorted(base.glob("*.safetensors")) if base.exists() else []
    hasher = hashlib.sha256()
    hasher.update(resolved.encode("utf-8"))
    for path in files:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return {
        "source_data_dir": resolved,
        "source_num_files": len(files),
        "source_fingerprint": hasher.hexdigest(),
    }


def export_saelens_cache(
    data_dir: str,
    layer: int,
    tr_cfg: dict,
    cache_dir: str,
    test_split: float = 0.2,
    seed: int = 42,
    shard_size: int = 200_000,
    rebuild: bool = False,
    filter_mode: str | None = None,
) -> dict[str, Any]:
    """Materialize cached activations into sharded .pt tensors for SAELens on-disk datasets."""
    out_dir = ensure_dir(cache_dir)
    marker = out_dir / "manifest.json"
    requested_filter_mode = filter_mode or "all"
    source_identity = build_source_data_identity(data_dir)
    if marker.exists() and not rebuild:
        with marker.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if (
            int(manifest.get("layer", -1)) == int(layer)
            and str(manifest.get("filter_mode", "all")) == requested_filter_mode
            and abs(float(manifest.get("test_split", -1.0)) - float(test_split)) < 1e-12
            and int(manifest.get("seed", -1)) == int(seed)
            and str(manifest.get("source_data_dir", "")) == source_identity["source_data_dir"]
            and str(manifest.get("source_fingerprint", "")) == source_identity["source_fingerprint"]
            and int(manifest.get("source_num_files", -1)) == source_identity["source_num_files"]
        ):
            return manifest

    for existing in out_dir.glob("*.pt"):
        existing.unlink(missing_ok=True)

    dataset = FlattenedActivationDataset(
        data_dir=data_dir,
        layer=layer,
        split="train",
        test_split=test_split,
        seed=seed,
        filter_mode=filter_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(tr_cfg.get("batch_size", 4096)),
        shuffle=False,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    shards = []
    buffer = []
    total_vectors = 0
    shard_idx = 0
    d_in = 0

    for batch in tqdm(loader, desc=f"Export SAELens cache L{layer}"):
        vecs = batch["activation"].to(torch.float32).cpu()
        if d_in == 0:
            d_in = int(vecs.shape[-1])
        buffer.append(vecs)
        current = sum(x.shape[0] for x in buffer)
        if current >= shard_size:
            shard = torch.cat(buffer, dim=0)
            shard_path = out_dir / f"acts_layer{layer}_shard{shard_idx:05d}.pt"
            torch.save(shard, shard_path)
            shards.append(str(shard_path))
            total_vectors += int(shard.shape[0])
            shard_idx += 1
            buffer = []

    if buffer:
        shard = torch.cat(buffer, dim=0)
        shard_path = out_dir / f"acts_layer{layer}_shard{shard_idx:05d}.pt"
        torch.save(shard, shard_path)
        shards.append(str(shard_path))
        total_vectors += int(shard.shape[0])

    manifest = {
        "layer": int(layer),
        "filter_mode": requested_filter_mode,
        "test_split": float(test_split),
        "seed": int(seed),
        "num_shards": len(shards),
        "num_vectors": int(total_vectors),
        "d_in": int(d_in) if d_in > 0 else 4096,
        "shards": shards,
    }
    manifest.update(source_identity)
    with marker.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def resolve_saelens_runner_config_kwargs(config_cls, candidate_kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(config_cls)
    out = {}
    for key, value in candidate_kwargs.items():
        if key in sig.parameters:
            out[key] = value
    return out


def validate_cached_activation_store(
    data_dir: str,
    layer: int,
    tr_cfg: dict,
    test_split: float,
    filter_mode: str | None = None,
) -> dict[str, Any]:
    batch_size = int(tr_cfg.get("batch_size", 4096))
    store = CachedActivationsStore(
        data_dir=data_dir,
        layer=layer,
        config={
            "batch_size": batch_size,
            "num_workers": tr_cfg.get("num_workers", 4),
            "test_split": test_split,
            "seed": tr_cfg.get("seed", 42),
        },
        split="train",
        filter_mode=filter_mode,
    )
    num_vectors = len(store)
    effective_train_batches = num_vectors // max(batch_size, 1)
    first_batch_shape: list[int] | None = None

    # CachedActivationsStore uses drop_last=True, so tiny datasets can yield 0 batches.
    if num_vectors > 0:
        if effective_train_batches > 0:
            try:
                first = next(iter(store))
                first_batch_shape = list(first.shape)
            except StopIteration:
                first_batch_shape = None
        else:
            # Probe with drop_last=False to surface the underlying sample shape for diagnostics.
            probe_loader = DataLoader(
                store.dataset,
                batch_size=min(batch_size, num_vectors),
                shuffle=False,
                num_workers=int(tr_cfg.get("num_workers", 4)),
                pin_memory=False,
                drop_last=False,
            )
            try:
                probe = next(iter(probe_loader))
                first_batch_shape = list(probe["activation"].shape)
            except StopIteration:
                first_batch_shape = None

    return {
        "num_vectors": num_vectors,
        "batch_size": batch_size,
        "effective_train_batches": effective_train_batches,
        "first_batch_shape": first_batch_shape,
    }


def train_manual(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    sae_cfg = resolve_sae_block(cfg)
    tr_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(int(tr_cfg.get("seed", 42)))

    dataset = FlattenedActivationDataset(
        data_dir=args.data_dir,
        layer=args.layer,
        split="train",
        test_split=float(args.test_split),
        seed=int(tr_cfg.get("seed", 42)),
        filter_mode=args.filter_mode,
    )

    configured_batch_size = int(tr_cfg.get("batch_size", 4096))
    if len(dataset) == 0:
        raise ValueError(
            "No training activation vectors available after split/filter. "
            "Adjust --filter_mode, test split, or rollout data."
        )
    effective_batch_size = min(configured_batch_size, len(dataset))
    drop_last = len(dataset) >= configured_batch_size
    if not drop_last:
        LOGGER.warning(
            "Training dataset (%d vectors) is smaller than batch size (%d). "
            "Using batch_size=%d with drop_last=False.",
            len(dataset),
            configured_batch_size,
            effective_batch_size,
        )

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=drop_last,
    )

    model = BatchTopKSAE(
        d_in=int(sae_cfg.get("d_in", 4096)),
        d_sae=int(sae_cfg.get("d_sae", 32768)),
        k=int(sae_cfg.get("k", 48)),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(tr_cfg.get("lr", 5e-5)))
    total_tokens = int(tr_cfg.get("total_training_tokens", 200_000_000))
    batch_size = effective_batch_size
    total_steps = max(total_tokens // max(batch_size, 1), 1)
    scheduler = build_scheduler(optimizer, total_steps)

    ckpt_dir = ensure_dir(args.checkpoint_dir)
    run_name = f"manual_layer{args.layer}"
    use_wandb = maybe_init_wandb(tr_cfg, run_name)
    wandb_mod = None
    if use_wandb:
        import wandb as wandb_mod  # type: ignore

    start_step = 0
    if args.resume and args.resume_path:
        ckpt = load_checkpoint(args.resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        LOGGER.info("Resumed from %s at step %d", args.resume_path, start_step)

    grad_clip = float(tr_cfg.get("max_grad_norm", 1.0))
    checkpoint_interval = int(tr_cfg.get("checkpoint_interval", 10000))
    eval_interval = int(tr_cfg.get("eval_interval", 5000))

    model.train()
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc=f"Train SAE L{args.layer}")

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            x = batch["activation"].to(device=device, dtype=torch.float32)
            x = normalize_expected_average_only_in(x)

            optimizer.zero_grad(set_to_none=True)
            recon, acts = model(x)
            recon_loss = (x - recon).pow(2).mean()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                model.decoder.weight.data = nn.functional.normalize(model.decoder.weight.data, dim=0)

            step += 1
            pbar.update(1)

            if step % 100 == 0:
                metrics = {
                    "step": step,
                    "recon_loss": float(recon_loss.item()),
                    "l0": compute_l0(acts),
                    "dead_features": compute_dead_features(acts),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                if use_wandb:
                    wandb_mod.log(metrics)

            if step % eval_interval == 0:
                with torch.no_grad():
                    eval_fvu = compute_fvu(x, recon)
                if use_wandb:
                    wandb_mod.log({"step": step, "fvu": eval_fvu})

            if step % checkpoint_interval == 0:
                ckpt_path = ckpt_dir / f"sae_layer{args.layer}_step{step}.pt"
                save_checkpoint(
                    ckpt_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "layer": args.layer,
                        "config": cfg,
                    },
                )

    pbar.close()

    final_path = ckpt_dir / f"sae_layer{args.layer}_final.pt"
    save_checkpoint(
        final_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "layer": args.layer,
            "config": cfg,
        },
    )

    if use_wandb:
        wandb_mod.finish()

    return {
        "checkpoint": str(final_path),
        "steps": step,
        "backend": "manual",
    }


def train_saelens(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Attempt SAELens runner with exported on-disk cached activations."""

    try:
        from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    except Exception as exc:
        LOGGER.warning("Failed importing SAELens runner (%s). Falling back to manual.", exc)
        return train_manual(args, cfg)

    sae_cfg = resolve_sae_block(cfg)
    tr_cfg = cfg["training"]

    store_info = validate_cached_activation_store(
        data_dir=args.data_dir,
        layer=args.layer,
        tr_cfg=tr_cfg,
        test_split=args.test_split,
        filter_mode=args.filter_mode,
    )
    LOGGER.info("CachedActivationsStore validation: %s", store_info)
    if int(store_info.get("effective_train_batches", 0)) <= 0:
        LOGGER.warning(
            "Cached activations produce 0 full batches for SAELens "
            "(num_vectors=%s batch_size=%s filter_mode=%s). Falling back to manual backend.",
            store_info.get("num_vectors"),
            store_info.get("batch_size"),
            args.filter_mode or "all",
        )
        return train_manual(args, cfg)

    ckpt_dir = ensure_dir(args.checkpoint_dir)
    cache_tag = args.filter_mode or "all"
    cache_dir = args.saelens_cache_dir or str(ckpt_dir / f"saelens_cache_layer{args.layer}_{cache_tag}")
    cache_manifest = export_saelens_cache(
        data_dir=args.data_dir,
        layer=args.layer,
        tr_cfg=tr_cfg,
        cache_dir=cache_dir,
        test_split=args.test_split,
        seed=int(tr_cfg.get("seed", 42)),
        shard_size=int(args.saelens_shard_size),
        rebuild=bool(args.saelens_rebuild_cache),
        filter_mode=args.filter_mode,
    )
    LOGGER.info("SAELens cache manifest: %s", cache_manifest)

    if args.saelens_validate_only:
        return {
            "backend": "saelens",
            "cache_manifest": cache_manifest,
            "note": "Validated cache and store only; training not started (--saelens_validate_only).",
        }

    try:
        candidate_kwargs = dict(
            model_name="openvla-cached",
            hook_name=f"blocks.{args.layer}.hook_resid_post",
            d_in=int(sae_cfg.get("d_in", 4096)),
            architecture=str(sae_cfg.get("architecture", "batch_topk")),
            d_sae=int(sae_cfg.get("d_sae", 32768)),
            k=int(sae_cfg.get("k", 48)),
            lr=float(tr_cfg.get("lr", 5e-5)),
            train_batch_size_tokens=int(tr_cfg.get("batch_size", 4096)),
            training_tokens=int(tr_cfg.get("total_training_tokens", 200_000_000)),
            lr_warm_up_steps=int(tr_cfg.get("lr_warmup_steps", 1000)),
            normalize_activations=str(tr_cfg.get("normalize_activations", "expected_average_only_in")),
            checkpoint_path=str(ckpt_dir),
            wandb_project=str(tr_cfg.get("wandb_project", "safesae-vla")),
            log_to_wandb=bool(tr_cfg.get("log_to_wandb", True)),
            dataset_path=cache_dir,
            is_dataset_on_disk=True,
            use_cached_activations=True,
            cached_activations_path=cache_dir,
        )
        cfg_kwargs = resolve_saelens_runner_config_kwargs(LanguageModelSAERunnerConfig, candidate_kwargs)
        if "dataset_path" not in cfg_kwargs and "cached_activations_path" not in cfg_kwargs:
            raise RuntimeError(
                "SAELens version does not expose on-disk dataset args in LanguageModelSAERunnerConfig."
            )

        runner_cfg = LanguageModelSAERunnerConfig(**cfg_kwargs)

        runner = SAETrainingRunner(runner_cfg)
        runner.run()

        return {
            "checkpoint_dir": str(ckpt_dir),
            "cache_manifest": cache_manifest,
            "backend": "saelens",
            "note": "SAELens runner completed.",
        }

    except Exception as exc:
        LOGGER.warning("SAELens training failed (%s). Falling back to manual backend.", exc, exc_info=True)
        return train_manual(args, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE on cached OpenVLA activations")
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--data_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--backend", type=str, default="saelens", choices=["saelens", "manual"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--filter_mode", type=str, default=None, choices=["safe", "unsafe"])
    parser.add_argument("--saelens_cache_dir", type=str, default="")
    parser.add_argument("--saelens_rebuild_cache", action="store_true")
    parser.add_argument("--saelens_validate_only", action="store_true")
    parser.add_argument("--saelens_shard_size", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_yaml(args.config)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.backend == "manual":
        result = train_manual(args, cfg)
    else:
        result = train_saelens(args, cfg)

    LOGGER.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
