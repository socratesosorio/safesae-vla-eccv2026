"""BatchTopK SAE architecture used by the refactored training pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BatchTopKSAE(nn.Module):
    """
    Batch Top-K sparse autoencoder.

    Args:
        d_in: input dimension.
        d_sae: dictionary width.
        k: number of retained features per sample.
    """

    def __init__(self, d_in: int = 4096, d_sae: int = 16384, k: int = 32):
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.k = int(k)

        self.b_pre = nn.Parameter(torch.zeros(self.d_in))
        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_in))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_pre
        pre_acts = x_centered @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
        topk_vals = torch.relu(topk_vals)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        recon, acts = self.forward(x)
        loss = (x - recon).pow(2).sum(dim=-1).mean()

        x_centered = x - x.mean(dim=0, keepdim=True)
        total_var = x_centered.pow(2).sum().clamp_min(1e-8)
        resid_var = (x - recon).pow(2).sum()
        fvu = float((resid_var / total_var).item())

        non_zero = acts > 0
        mean_activation = float(acts[non_zero].mean().item()) if non_zero.any() else 0.0
        metrics = {
            "loss": float(loss.item()),
            "l0": float(non_zero.float().sum(dim=-1).mean().item()),
            "fvu": fvu,
            "dead_features_pct": float((acts.sum(dim=0) == 0).float().mean().item() * 100.0),
            "mean_activation": mean_activation,
        }
        return loss, metrics
