"""Metric helpers for safety analysis and monitor evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float


def rate(events: Iterable[bool | int]) -> float:
    arr = np.asarray(list(events), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def bootstrap_rate_ci(events: Iterable[bool | int], n_boot: int = 1000, alpha: float = 0.05) -> BootstrapCI:
    arr = np.asarray(list(events), dtype=np.float32)
    if arr.size == 0:
        return BootstrapCI(mean=0.0, lo=0.0, hi=0.0)

    rng = np.random.default_rng(42)
    samples = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(arr[idx].mean())

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapCI(mean=float(arr.mean()), lo=lo, hi=hi)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def cost_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, fn_weight: float = 10.0) -> float:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    beta2 = fn_weight
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return non-dominated points for (success_rate, -safety_violation_rate)."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p[0], reverse=True)
    frontier: list[tuple[float, float]] = []
    best_second = -np.inf
    for sr, neg_svr in sorted_pts:
        if neg_svr > best_second:
            frontier.append((sr, neg_svr))
            best_second = neg_svr
    return frontier
