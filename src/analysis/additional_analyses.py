"""Additional analyses for ECCV paper: sparsity curves, temporal patterns, per-suite breakdown, feature inspection, and latency benchmarks."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SAFETY_CATEGORIES = [
    "collision",
    "excessive_force",
    "boundary_violation",
    "high_approach_speed",
    "object_drop",
]

DEFAULT_K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _load_top_features(feature_csv: str, n: int) -> np.ndarray:
    """Return top-n feature indices ranked by composite_score."""
    df = pd.read_csv(feature_csv)
    return df["feature_idx"].head(n).astype(int).values


def sparsity_performance_curve(
    x_sae: np.ndarray,
    y_any: np.ndarray,
    y_cat: np.ndarray,
    feature_csv: str,
    k_values: Sequence[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Train LR monitors on top-k SAE features and report AUROC vs sparsity."""
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    df = pd.read_csv(feature_csv)
    ranked_features = df["feature_idx"].astype(int).values
    max_k = max(k_values)
    # Cap to number of available features
    ranked_features = ranked_features[: min(max_k, len(ranked_features))]

    # Train/test split
    idx = np.arange(len(y_any))
    try:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y_any)
    except ValueError:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)

    rows = []
    for k in k_values:
        k_actual = min(k, len(ranked_features))
        if k_actual == 0:
            continue
        top_k_idx = ranked_features[:k_actual]

        # Mask: zero out all columns except top-k
        x_masked = np.zeros_like(x_sae)
        x_masked[:, top_k_idx] = x_sae[:, top_k_idx]

        x_train = x_masked[train_idx]
        x_test = x_masked[test_idx]
        y_train = y_any[train_idx]
        y_test = y_any[test_idx]

        if len(np.unique(y_train)) < 2:
            auroc_overall = 0.5
            scores = np.full(len(y_test), 0.5)
        else:
            model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_test)[:, 1]
            auroc_overall = _safe_auroc(y_test, scores)

        row = {"k": k, "k_actual": k_actual, "auroc_overall": auroc_overall}

        # Per-category AUROC
        for i, cat in enumerate(SAFETY_CATEGORIES):
            y_cat_test = y_cat[test_idx, i]
            row[f"auroc_{cat}"] = _safe_auroc(y_cat_test, scores)

        rows.append(row)

    return pd.DataFrame(rows)


def temporal_violation_patterns(
    x_sae: np.ndarray,
    y_cat: np.ndarray,
    ep_idx_per_step: np.ndarray,
    feature_csv: str,
    window: int = 5,
    top_k: int = 20,
) -> pd.DataFrame:
    """Compute mean activation of top features around violation onsets."""
    top_features = _load_top_features(feature_csv, top_k)
    if len(top_features) == 0:
        return pd.DataFrame(columns=["category", "relative_t", "mean_activation", "std_activation", "n_events"])

    # Extract top-k feature activations
    x_top = x_sae[:, top_features]  # [steps, top_k]
    x_mean = x_top.mean(axis=1)  # [steps] — average across top features

    rows = []
    for cat_idx, cat in enumerate(SAFETY_CATEGORIES):
        y_c = y_cat[:, cat_idx]

        # Find violation onsets: first step of each consecutive run, respecting episode boundaries
        onsets = []
        for ep in np.unique(ep_idx_per_step):
            ep_mask = ep_idx_per_step == ep
            ep_steps = np.where(ep_mask)[0]
            if len(ep_steps) == 0:
                continue
            ep_labels = y_c[ep_steps]
            for j in range(len(ep_labels)):
                if ep_labels[j] == 1 and (j == 0 or ep_labels[j - 1] == 0):
                    onsets.append(ep_steps[j])

        if len(onsets) == 0:
            continue

        # Collect activation windows around each onset
        for rel_t in range(-window, window + 1):
            vals = []
            for onset in onsets:
                t = onset + rel_t
                if 0 <= t < len(x_mean):
                    # Only include if same episode
                    if ep_idx_per_step[t] == ep_idx_per_step[onset]:
                        vals.append(x_mean[t])
            if vals:
                rows.append({
                    "category": cat,
                    "relative_t": rel_t,
                    "mean_activation": float(np.mean(vals)),
                    "std_activation": float(np.std(vals)),
                    "n_events": len(vals),
                })

    return pd.DataFrame(rows)


def per_suite_breakdown(
    x_sae: np.ndarray,
    y_any: np.ndarray,
    y_cat: np.ndarray,
    ep_idx_per_step: np.ndarray,
    metadata: list[dict],
    feature_csv: str,
    top_k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute per-suite violation rates, feature activations, and monitor AUROC."""
    top_features = _load_top_features(feature_csv, top_k)

    # Map episode idx -> suite
    ep_suite = {}
    for ep_idx, meta in enumerate(metadata):
        ep_suite[ep_idx] = str(meta.get("suite", "unknown"))

    # Assign suite to each step
    step_suite = np.array([ep_suite.get(int(ep), "unknown") for ep in ep_idx_per_step])
    suites = sorted(set(step_suite))

    rows = []
    for suite in suites:
        mask = step_suite == suite
        n_steps = int(mask.sum())

        # Count episodes in this suite
        eps_in_suite = set(ep_idx_per_step[mask].tolist())
        n_episodes = len(eps_in_suite)

        row = {"suite": suite, "n_episodes": n_episodes, "n_steps": n_steps}

        # Per-category violation rate (fraction of steps)
        for i, cat in enumerate(SAFETY_CATEGORIES):
            rate = float(y_cat[mask, i].mean()) if n_steps > 0 else 0.0
            row[f"{cat}_rate"] = rate

        # Mean top-k feature activation
        if len(top_features) > 0 and n_steps > 0:
            row["mean_top_k_activation"] = float(x_sae[mask][:, top_features].mean())
        else:
            row["mean_top_k_activation"] = 0.0

        # Per-suite AUROC (train on this suite's data with stratified split)
        x_suite = x_sae[mask]
        y_suite = y_any[mask]
        if len(np.unique(y_suite)) >= 2 and n_steps >= 20:
            try:
                si = np.arange(n_steps)
                tr, te = train_test_split(si, test_size=0.3, random_state=seed, stratify=y_suite)
                lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
                lr.fit(x_suite[tr], y_suite[tr])
                scores = lr.predict_proba(x_suite[te])[:, 1]
                row["auroc"] = _safe_auroc(y_suite[te], scores)
            except ValueError:
                row["auroc"] = 0.5
        else:
            row["auroc"] = 0.5

        rows.append(row)

    return pd.DataFrame(rows)


def feature_inspection_table(
    x_sae: np.ndarray,
    y_cat: np.ndarray,
    ep_idx_per_step: np.ndarray,
    feature_csv: str,
    top_k: int = 10,
    window: int = 5,
) -> pd.DataFrame:
    """Build a detailed inspection table for the top-k differential features."""
    df = pd.read_csv(feature_csv)
    top_features = df["feature_idx"].head(top_k).astype(int).values

    # y_any from y_cat
    y_any = y_cat.any(axis=1).astype(int)

    rows = []
    for rank, feat_idx in enumerate(top_features):
        feat_vals = x_sae[:, feat_idx]

        # Sparsity: fraction of steps where feature > 0
        sparsity = float((feat_vals > 0).mean())

        # Mean activation in violation vs safe steps
        viol_mask = y_any == 1
        safe_mask = y_any == 0
        mean_viol = float(feat_vals[viol_mask].mean()) if viol_mask.any() else 0.0
        mean_safe = float(feat_vals[safe_mask].mean()) if safe_mask.any() else 0.0

        # Point-biserial correlation with each category
        best_cat = "none"
        best_r = 0.0
        for i, cat in enumerate(SAFETY_CATEGORIES):
            y_c = y_cat[:, i]
            if len(np.unique(y_c)) < 2:
                continue
            try:
                r, _ = pointbiserialr(y_c, feat_vals)
                if abs(r) > abs(best_r):
                    best_r = float(r)
                    best_cat = cat
            except Exception:
                continue

        # Temporal profile: does the feature peak before, during, or after violations?
        # Find violation onsets and compute mean activation at each relative position
        temporal_peak = "unknown"
        onset_activations = {t: [] for t in range(-window, window + 1)}
        for ep in np.unique(ep_idx_per_step):
            ep_mask = ep_idx_per_step == ep
            ep_steps = np.where(ep_mask)[0]
            ep_labels = y_any[ep_steps]
            for j in range(len(ep_labels)):
                if ep_labels[j] == 1 and (j == 0 or ep_labels[j - 1] == 0):
                    onset = ep_steps[j]
                    for rel_t in range(-window, window + 1):
                        t = onset + rel_t
                        if 0 <= t < len(feat_vals) and ep_idx_per_step[t] == ep:
                            onset_activations[rel_t].append(feat_vals[t])

        mean_by_t = {}
        for rel_t, vals in onset_activations.items():
            if vals:
                mean_by_t[rel_t] = float(np.mean(vals))
        if mean_by_t:
            peak_t = max(mean_by_t, key=mean_by_t.get)
            if peak_t < 0:
                temporal_peak = "before"
            elif peak_t == 0:
                temporal_peak = "during"
            else:
                temporal_peak = "after"

        # Pull composite_score from the CSV if available
        feat_row = df[df["feature_idx"] == int(feat_idx)]
        composite = float(feat_row["composite_score"].iloc[0]) if not feat_row.empty and "composite_score" in feat_row.columns else 0.0

        rows.append({
            "rank": rank + 1,
            "feature_idx": int(feat_idx),
            "composite_score": composite,
            "top_category": best_cat,
            "point_biserial_r": best_r,
            "sparsity": sparsity,
            "mean_viol": mean_viol,
            "mean_safe": mean_safe,
            "viol_safe_ratio": mean_viol / max(mean_safe, 1e-8),
            "temporal_peak": temporal_peak,
        })

    return pd.DataFrame(rows)


def latency_report(
    x_sae: np.ndarray,
    sae_model: torch.nn.Module,
    batch_sizes: Sequence[int] | None = None,
    n_iters: int = 100,
) -> pd.DataFrame:
    """Benchmark SAE encode + LR predict latency across batch sizes and devices."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    d_in = x_sae.shape[1]
    rows = []

    # Fit a quick LR for timing
    # Use random labels just for timing purposes
    rng = np.random.RandomState(42)
    y_dummy = rng.randint(0, 2, size=min(1000, len(x_sae)))
    x_dummy = x_sae[:len(y_dummy)]
    lr = LogisticRegression(max_iter=100, C=0.1)
    if len(np.unique(y_dummy)) >= 2:
        lr.fit(x_dummy, y_dummy)
    else:
        lr.fit(x_dummy, np.concatenate([np.zeros(len(y_dummy) - 1), np.ones(1)]).astype(int))

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        device = torch.device(device_name)
        model = sae_model.to(device)
        model.eval()

        for bs in batch_sizes:
            # SAE encode timing
            x_batch = torch.randn(bs, d_in, device=device)
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model.encode(x_batch)
            if device_name == "cuda":
                torch.cuda.synchronize()

            encode_times = []
            for _ in range(n_iters):
                if device_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.encode(x_batch)
                if device_name == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                encode_times.append((t1 - t0) * 1000.0)

            rows.append({
                "method": "sae_encode",
                "batch_size": bs,
                "device": device_name,
                "mean_ms": float(np.mean(encode_times)),
                "std_ms": float(np.std(encode_times)),
            })

            # LR predict timing (always CPU)
            x_lr = np.random.randn(bs, x_sae.shape[1]).astype(np.float32)
            # Warmup
            for _ in range(5):
                lr.predict_proba(x_lr)

            lr_times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                lr.predict_proba(x_lr)
                t1 = time.perf_counter()
                lr_times.append((t1 - t0) * 1000.0)

            rows.append({
                "method": "lr_predict",
                "batch_size": bs,
                "device": "cpu",
                "mean_ms": float(np.mean(lr_times)),
                "std_ms": float(np.std(lr_times)),
            })

    # Move model back to original device (cpu by default)
    sae_model.to(torch.device("cpu"))

    return pd.DataFrame(rows)
