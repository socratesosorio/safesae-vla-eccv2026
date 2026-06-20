"""Plot utilities for safety feature visualization and analysis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "DejaVu Serif"


def plot_volcano(df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    x = df["effect_size"].to_numpy()
    y = -np.log10(df["adjusted_p"].to_numpy() + 1e-300)
    colors = np.where(df["significant"].to_numpy(), "tab:red", "tab:blue")
    ax.scatter(x, y, c=colors, s=10, alpha=0.7)
    ax.set_xlabel("Effect Size (Rank-Biserial Correlation)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Differential Activation Volcano Plot")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_heatmap(
    feature_traces: np.ndarray,
    violation_steps: list[int],
    output_path: str,
    title: str = "Safety Feature Activations",
) -> None:
    # feature_traces: [num_features, timesteps]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    sns.heatmap(feature_traces, cmap="mako", ax=ax)
    for step in violation_steps:
        ax.axvline(step, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feature Index (top-k)")
    ax.set_title(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, output_path: str) -> None:
    pivot = overlap_df.pivot(index="cat_a", columns="cat_b", values="jaccard").fillna(0.0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(pivot, annot=True, cmap="crest", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Top-Feature Category Overlap (Jaccard)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tsne(features: np.ndarray, labels: np.ndarray, output_path: str, title: str = "SAE Feature Space") -> None:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=12, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Label")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def extract_top_feature_traces(
    sae,
    episode_activations: torch.Tensor,
    top_feature_indices: list[int],
) -> np.ndarray:
    with torch.no_grad():
        # [steps, 7, 4096] -> [steps, 4096]
        pooled = episode_activations.mean(dim=1)
        feats = sae.encode(pooled.to(next(sae.parameters()).device))
        traces = feats[:, top_feature_indices].transpose(0, 1).cpu().numpy()
    return traces
