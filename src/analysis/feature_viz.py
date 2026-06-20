"""Feature visualization helpers (refactor-friendly alias module)."""

from __future__ import annotations

from src.analysis.feature_visualization import (  # re-export
    extract_top_feature_traces,
    plot_feature_heatmap,
    plot_overlap_heatmap,
    plot_tsne,
    plot_volcano,
)

__all__ = [
    "plot_volcano",
    "plot_feature_heatmap",
    "plot_overlap_heatmap",
    "plot_tsne",
    "extract_top_feature_traces",
]
