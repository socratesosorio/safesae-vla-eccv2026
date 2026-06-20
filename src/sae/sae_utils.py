"""SAE utility functions for evaluation and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import torch


def normalize_expected_average_only_in(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True)
    mean_norm = norms.mean().clamp_min(eps)
    return x / mean_norm


def compute_fvu(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> float:
    resid_var = torch.var(x - x_hat)
    total_var = torch.var(x)
    return float((resid_var / (total_var + eps)).item())


def compute_l0(acts: torch.Tensor) -> float:
    return float((acts > 0).float().sum(dim=-1).mean().item())


def compute_dead_features(acts: torch.Tensor) -> float:
    dead = (acts.sum(dim=0) <= 0).float().mean().item()
    return float(dead)


def save_checkpoint(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
