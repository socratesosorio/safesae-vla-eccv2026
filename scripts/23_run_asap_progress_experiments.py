"""Run ASAP workshop stress tests for progress SAE features.

This script is intentionally offline: it uses cached rollout tensors, cached
SAE checkpoints, and cached progress labels.  It covers:

1. Feature-space high-progress -> low-progress patching.
2. Ablation specificity against random and bottom-ranked active features.
3. Dose-response for top progress features.
4. Progress proxy validation against telemetry and success metadata.
5. Top-feature stability under alternative progress proxies.
6. Safety threshold sensitivity from cached telemetry relabeling.

The action-shift measurement is a cached-action readout proxy: a Ridge readout
is fit from SAE features to stored continuous actions, then interventions are
measured as predicted action deltas.  The script also reports SAE decoder-space
activation deltas so the intervention magnitude is transparent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint  # noqa: E402
from src.data.safety_labeler import SAFETY_CATEGORIES, SafetyLabeler  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


@dataclass
class SampleBundle:
    raw_activations: np.ndarray
    features: np.ndarray
    actions: np.ndarray
    labels: np.ndarray
    progress_norm: np.ndarray
    episode_ids: np.ndarray
    stages: np.ndarray
    final_distance: np.ndarray
    final_displacement: np.ndarray
    path_length: np.ndarray
    episode_success: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--labels_full_csv", type=str, required=True)
    p.add_argument("--top_features_csv", type=str, required=True)
    p.add_argument("--sae_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="logs/asap_workshop_experiments")
    p.add_argument("--rollout_config", type=str, default="", help="Optional rollout config used as the 1.0x safety threshold baseline")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--max_timesteps_per_episode", type=int, default=8)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random_trials", type=int, default=20)
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan")
    rho, p_val = spearmanr(x[mask], y[mask])
    return float(rho), float(p_val)


def episode_success_from_meta(meta_path: Path, tensors: dict[str, np.ndarray]) -> bool:
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "episode_success" in meta:
                return bool(meta["episode_success"])
            if "success" in meta:
                return bool(meta["success"])
        except json.JSONDecodeError:
            pass
    if "episode_success" in tensors:
        return bool(np.asarray(tensors["episode_success"]).astype(bool).any())
    return False


def sampled_indices(num_steps: int, max_steps: int) -> np.ndarray:
    if num_steps <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_steps <= max_steps:
        return np.arange(num_steps, dtype=np.int64)
    return np.unique(np.linspace(0, num_steps - 1, num=max_steps, dtype=np.int64))


@torch.no_grad()
def encode_numpy(
    sae: torch.nn.Module,
    acts: np.ndarray,
    norm_factor: float,
    device: torch.device,
    chunk: int = 1024,
) -> np.ndarray:
    x = torch.from_numpy(acts.astype(np.float32, copy=False)) / float(max(norm_factor, 1e-8))
    outputs: list[torch.Tensor] = []
    for start in range(0, x.shape[0], chunk):
        feats = sae.encode(x[start : start + chunk].to(device)).detach().cpu()
        outputs.append(feats)
    return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def load_samples(
    *,
    data_dir: Path,
    labels_full: pd.DataFrame,
    layer: int,
    sae: torch.nn.Module,
    norm_factor: float,
    device: torch.device,
    max_timesteps_per_episode: int,
    max_episodes: int,
) -> SampleBundle:
    label_df = labels_full.set_index("episode_id")
    key = f"activations_layer{layer}"
    feature_blocks: list[np.ndarray] = []
    raw_blocks: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    labels: list[int] = []
    progress_vals: list[float] = []
    episode_ids: list[str] = []
    stages: list[float] = []
    final_distance: list[float] = []
    final_displacement: list[float] = []
    path_length: list[float] = []
    successes: list[int] = []

    tensor_paths = sorted(data_dir.rglob("rollout_*.safetensors"))
    if max_episodes > 0:
        tensor_paths = tensor_paths[: int(max_episodes)]
    for tensor_path in tensor_paths:
        episode_id = tensor_path.stem
        if episode_id not in label_df.index:
            continue
        label = int(label_df.loc[episode_id, "label"])
        progress = float(label_df.loc[episode_id, "progress_norm"])
        if label not in (0, 1):
            continue

        tensors: dict[str, np.ndarray] = {}
        with safe_open(str(tensor_path), framework="np") as f:
            keys = set(f.keys())
            if key not in keys or "actions" not in keys or "eef_positions" not in keys:
                continue
            for name in (key, "actions", "eef_positions", "contact_forces", "episode_success"):
                if name in keys:
                    tensors[name] = f.get_tensor(name)

        acts = tensors[key].astype(np.float32)
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        eef = np.asarray(tensors["eef_positions"], dtype=np.float32)
        actions = np.asarray(tensors["actions"], dtype=np.float32)
        n_steps = min(step_vecs.shape[0], actions.shape[0], eef.shape[0])
        if n_steps <= 1:
            continue
        idx = sampled_indices(n_steps, max_timesteps_per_episode)
        raw_sampled = step_vecs[idx].astype(np.float32, copy=False)
        feats = encode_numpy(sae, raw_sampled, norm_factor=norm_factor, device=device)
        raw_blocks.append(raw_sampled)
        feature_blocks.append(feats)
        action_blocks.append(actions[idx])

        disp = float(np.linalg.norm(eef[-1] - eef[0]))
        deltas = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        path_len = float(deltas.sum())
        final_dist = -float(label_df.loc[episode_id, "progress_raw"])
        if not np.isfinite(final_dist):
            final_dist = disp
        success = int(episode_success_from_meta(tensor_path.with_suffix(".json"), tensors))

        labels.extend([label] * len(idx))
        progress_vals.extend([progress] * len(idx))
        episode_ids.extend([episode_id] * len(idx))
        stages.extend((idx / max(n_steps - 1, 1)).astype(float).tolist())
        final_distance.extend([final_dist] * len(idx))
        final_displacement.extend([disp] * len(idx))
        path_length.extend([path_len] * len(idx))
        successes.extend([success] * len(idx))

    if not feature_blocks:
        raise RuntimeError(f"No usable samples found under {data_dir}")

    return SampleBundle(
        raw_activations=np.concatenate(raw_blocks, axis=0).astype(np.float32, copy=False),
        features=np.concatenate(feature_blocks, axis=0),
        actions=np.concatenate(action_blocks, axis=0).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        progress_norm=np.asarray(progress_vals, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=object),
        stages=np.asarray(stages, dtype=np.float32),
        final_distance=np.asarray(final_distance, dtype=np.float32),
        final_displacement=np.asarray(final_displacement, dtype=np.float32),
        path_length=np.asarray(path_length, dtype=np.float32),
        episode_success=np.asarray(successes, dtype=np.int64),
    )


def select_feature_sets(
    features: np.ndarray,
    top_features: list[int],
    top_k: int,
    seed: int,
    random_trials: int,
) -> dict[str, list[np.ndarray]]:
    active = np.flatnonzero((features > 0).any(axis=0))
    means = features.mean(axis=0)
    top = np.asarray(top_features[:top_k], dtype=np.int64)
    active_not_top = np.asarray([i for i in active if i not in set(top.tolist())], dtype=np.int64)
    bottom = active_not_top[np.argsort(means[active_not_top])[: len(top)]] if len(active_not_top) else active[: len(top)]
    rng = np.random.default_rng(seed)
    random_sets = []
    for _ in range(max(int(random_trials), 1)):
        if len(active_not_top) >= len(top):
            random_sets.append(rng.choice(active_not_top, size=len(top), replace=False))
        else:
            random_sets.append(rng.choice(active, size=len(top), replace=False))
    return {"top": [top], "bottom": [bottom], "random": random_sets, "active": [active]}


def fit_models(bundle: SampleBundle, active_cols: np.ndarray, seed: int):
    idx = np.arange(bundle.features.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=seed,
        stratify=bundle.labels if len(np.unique(bundle.labels)) > 1 else None,
    )
    x_train = bundle.features[train_idx][:, active_cols]
    x_test = bundle.features[test_idx][:, active_cols]
    scaler = StandardScaler(with_mean=False)
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    probe = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=seed)
    probe.fit(x_train_s, bundle.labels[train_idx])
    test_logits = probe.decision_function(x_test_s)
    auroc = safe_auroc(bundle.labels[test_idx], test_logits)

    raw_scaler = StandardScaler()
    action_readout = Ridge(alpha=10.0)
    raw_train = raw_scaler.fit_transform(bundle.raw_activations[train_idx])
    raw_test = raw_scaler.transform(bundle.raw_activations[test_idx])
    action_readout.fit(raw_train, bundle.actions[train_idx])
    pred = action_readout.predict(raw_test)
    denom = float(np.var(bundle.actions[test_idx], axis=0).sum())
    action_r2 = 1.0 - float(np.square(bundle.actions[test_idx] - pred).sum()) / max(
        float(np.square(bundle.actions[test_idx] - bundle.actions[test_idx].mean(axis=0)).sum()),
        1e-8,
    )
    return train_idx, test_idx, scaler, probe, raw_scaler, action_readout, {"progress_probe_auroc": auroc, "raw_activation_action_readout_r2": action_r2, "action_variance_sum": denom}


def model_outputs(
    x_full: np.ndarray,
    raw_x: np.ndarray,
    active_cols: np.ndarray,
    feature_scaler: StandardScaler,
    probe: LogisticRegression,
    raw_scaler: StandardScaler,
    action_readout: Ridge,
) -> tuple[np.ndarray, np.ndarray]:
    xs = feature_scaler.transform(x_full[:, active_cols])
    raw_s = raw_scaler.transform(raw_x)
    return probe.decision_function(xs), action_readout.predict(raw_s)


def patch_features(x: np.ndarray, indices: Iterable[int], values: np.ndarray | float) -> np.ndarray:
    y = x.copy()
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size:
        y[:, idx] = values
    return y


def decoded_delta(sae: torch.nn.Module, before: np.ndarray, after: np.ndarray, norm_factor: float) -> np.ndarray:
    delta = after - before
    changed = np.flatnonzero(np.abs(delta).sum(axis=0) > 0)
    if changed.size == 0:
        return np.zeros((before.shape[0], sae.d_in), dtype=np.float32)
    w_dec = sae.W_dec.detach().cpu().numpy().astype(np.float32, copy=False)
    return (delta[:, changed] @ w_dec[changed]) * float(norm_factor)


def summarize_intervention(
    name: str,
    before_x: np.ndarray,
    after_x: np.ndarray,
    before_raw: np.ndarray,
    active_cols: np.ndarray,
    feature_scaler: StandardScaler,
    probe: LogisticRegression,
    raw_scaler: StandardScaler,
    action_readout: Ridge,
    sae: torch.nn.Module,
    norm_factor: float,
) -> dict[str, float | str]:
    raw_delta = decoded_delta(sae, before_x, after_x, norm_factor=norm_factor)
    after_raw = before_raw + raw_delta
    before_logit, before_action = model_outputs(
        before_x, before_raw, active_cols, feature_scaler, probe, raw_scaler, action_readout
    )
    after_logit, after_action = model_outputs(
        after_x, after_raw, active_cols, feature_scaler, probe, raw_scaler, action_readout
    )
    action_shift = np.linalg.norm(after_action - before_action, axis=1)
    decoder_shift = np.linalg.norm(raw_delta, axis=1)
    return {
        "condition": name,
        "n_samples": int(before_x.shape[0]),
        "mean_progress_logit_delta": float(np.mean(after_logit - before_logit)),
        "median_progress_logit_delta": float(np.median(after_logit - before_logit)),
        "mean_raw_action_readout_shift_l2": float(np.mean(action_shift)),
        "median_raw_action_readout_shift_l2": float(np.median(action_shift)),
        "mean_raw_activation_delta_l2": float(np.mean(decoder_shift)),
        "median_raw_activation_delta_l2": float(np.median(decoder_shift)),
    }


def run_interventions(
    bundle: SampleBundle,
    top_features: list[int],
    sae: torch.nn.Module,
    norm_factor: float,
    seed: int,
    top_k: int,
    random_trials: int,
    output_dir: Path,
) -> dict[str, float]:
    sets = select_feature_sets(bundle.features, top_features, top_k=top_k, seed=seed, random_trials=random_trials)
    active_cols = sets["active"][0]
    train_idx, test_idx, scaler, probe, raw_scaler, action_readout, diagnostics = fit_models(bundle, active_cols, seed=seed)
    test_low = test_idx[bundle.labels[test_idx] == 0]
    if len(test_low) == 0:
        test_low = test_idx

    x_low = bundle.features[test_low]
    raw_low = bundle.raw_activations[test_low]
    high_mean = bundle.features[bundle.labels == 1].mean(axis=0)
    top = sets["top"][0]
    bottom = sets["bottom"][0]

    rows: list[dict[str, float | str]] = []
    rows.append(
        summarize_intervention(
            "patch_top20_high_progress_mean",
            x_low,
            patch_features(x_low, top, high_mean[top]),
            raw_low,
            active_cols,
            scaler,
            probe,
            raw_scaler,
            action_readout,
            sae,
            norm_factor,
        )
    )
    rows.append(
        summarize_intervention(
            "zero_top20_progress_features",
            x_low,
            patch_features(x_low, top, 0.0),
            raw_low,
            active_cols,
            scaler,
            probe,
            raw_scaler,
            action_readout,
            sae,
            norm_factor,
        )
    )
    rows.append(
        summarize_intervention(
            "zero_bottom20_active_features",
            x_low,
            patch_features(x_low, bottom, 0.0),
            raw_low,
            active_cols,
            scaler,
            probe,
            raw_scaler,
            action_readout,
            sae,
            norm_factor,
        )
    )
    random_rows = [
        summarize_intervention(
            f"zero_random20_active_features_trial{i:02d}",
            x_low,
            patch_features(x_low, idx, 0.0),
            raw_low,
            active_cols,
            scaler,
            probe,
            raw_scaler,
            action_readout,
            sae,
            norm_factor,
        )
        for i, idx in enumerate(sets["random"])
    ]
    rows.extend(random_rows)
    intervention_df = pd.DataFrame(rows)
    intervention_df.to_csv(output_dir / "feature_patching_and_ablation.csv", index=False)

    rand = pd.DataFrame(random_rows)
    collapsed = pd.concat(
        [
            intervention_df[~intervention_df["condition"].str.contains("trial", regex=False)],
            pd.DataFrame(
                [
                    {
                        "condition": "zero_random20_active_features_mean",
                        "n_samples": int(x_low.shape[0]),
                        **{
                            col: float(rand[col].mean())
                            for col in rand.columns
                            if col not in {"condition", "n_samples"}
                        },
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    collapsed.to_csv(output_dir / "feature_patching_and_ablation_summary.csv", index=False)

    dose_rows = []
    top5 = np.asarray(top_features[:5], dtype=np.int64)
    top5_high = high_mean[top5]
    baseline_logit, baseline_action = model_outputs(
        x_low, raw_low, active_cols, scaler, probe, raw_scaler, action_readout
    )
    for multiplier in [0.0, 0.5, 1.0, 1.5, 2.0]:
        patched = x_low.copy()
        patched[:, top5] = top5_high[None, :] * float(multiplier)
        raw_delta = decoded_delta(sae, x_low, patched, norm_factor=norm_factor)
        logit, action_pred = model_outputs(
            patched, raw_low + raw_delta, active_cols, scaler, probe, raw_scaler, action_readout
        )
        dose_rows.append(
            {
                "multiplier": float(multiplier),
                "mean_progress_logit": float(np.mean(logit)),
                "mean_progress_logit_delta": float(np.mean(logit - baseline_logit)),
                "mean_raw_action_readout_shift_l2": float(np.linalg.norm(action_pred - baseline_action, axis=1).mean()),
                "mean_raw_activation_delta_l2": float(np.linalg.norm(raw_delta, axis=1).mean()),
            }
        )
    dose_df = pd.DataFrame(dose_rows)
    dose_df.to_csv(output_dir / "dose_response_top5_progress_features.csv", index=False)
    rho, p_val = safe_spearman(dose_df["multiplier"].to_numpy(), dose_df["mean_progress_logit"].to_numpy())
    diagnostics.update(
        {
            "num_samples": int(bundle.features.shape[0]),
            "num_active_features": int(len(active_cols)),
            "num_low_test_samples": int(len(test_low)),
            "dose_response_spearman_rho": rho,
            "dose_response_spearman_p": p_val,
        }
    )
    return diagnostics


def episode_level_frame(bundle: SampleBundle) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "episode_id": bundle.episode_ids,
            "label": bundle.labels,
            "progress_norm": bundle.progress_norm,
            "stage": bundle.stages,
            "final_distance": bundle.final_distance,
            "final_displacement": bundle.final_displacement,
            "path_length": bundle.path_length,
            "episode_success": bundle.episode_success,
        }
    )
    return (
        df.groupby("episode_id", as_index=False)
        .agg(
            label=("label", "max"),
            progress_norm=("progress_norm", "mean"),
            mean_stage=("stage", "mean"),
            final_distance=("final_distance", "mean"),
            final_displacement=("final_displacement", "mean"),
            path_length=("path_length", "mean"),
            episode_success=("episode_success", "max"),
        )
        .reset_index(drop=True)
    )


def run_proxy_validation(bundle: SampleBundle, top_features: list[int], output_dir: Path) -> dict[str, float]:
    ep = episode_level_frame(bundle)
    rows = []
    for name in ["episode_success", "final_distance", "final_displacement", "path_length"]:
        rho, p_val = safe_spearman(ep["progress_norm"].to_numpy(), ep[name].to_numpy())
        rows.append({"proxy": name, "spearman_rho": rho, "p_value": p_val, "n_episodes": int(len(ep))})
    if len(np.unique(ep["episode_success"])) > 1:
        rows.append(
            {
                "proxy": "episode_success_auroc",
                "spearman_rho": safe_auroc(ep["episode_success"].to_numpy(), ep["progress_norm"].to_numpy()),
                "p_value": float("nan"),
                "n_episodes": int(len(ep)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "progress_proxy_validation.csv", index=False)

    feature_cols = [f"f{i}" for i in range(bundle.features.shape[1])]
    ep_features = pd.DataFrame(bundle.features, columns=feature_cols)
    ep_features.insert(0, "episode_id", bundle.episode_ids)
    ep_features = ep_features.groupby("episode_id", as_index=False).mean()
    ep_full = ep.merge(ep_features, on="episode_id", how="inner")

    alternatives = {
        "progress_norm": ep_full["progress_norm"].to_numpy(),
        "negative_final_distance": -ep_full["final_distance"].to_numpy(),
        "final_displacement": ep_full["final_displacement"].to_numpy(),
        "path_length": ep_full["path_length"].to_numpy(),
    }
    stability_rows = []
    top_set = set(int(x) for x in top_features[:20])
    feat_mat = ep_full[feature_cols].to_numpy(dtype=np.float32)
    for name, metric in alternatives.items():
        if len(np.unique(metric[np.isfinite(metric)])) < 4:
            continue
        q_low, q_high = np.nanquantile(metric, [0.25, 0.75])
        low = metric <= q_low
        high = metric >= q_high
        if int(low.sum()) == 0 or int(high.sum()) == 0:
            continue
        mean_delta = feat_mat[high].mean(axis=0) - feat_mat[low].mean(axis=0)
        ranked = np.argsort(np.abs(mean_delta))[::-1][:20]
        overlap = len(top_set.intersection(set(int(x) for x in ranked)))
        stability_rows.append(
            {
                "alternative_proxy": name,
                "top20_overlap_with_primary": int(overlap),
                "jaccard_with_primary_top20": float(overlap / max(len(set(ranked).union(top_set)), 1)),
                "top_features": " ".join(str(int(x)) for x in ranked[:10]),
            }
        )
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(output_dir / "alternative_progress_feature_stability.csv", index=False)
    mean_overlap = float(stability_df["top20_overlap_with_primary"].mean()) if not stability_df.empty else float("nan")
    return {"mean_alt_proxy_top20_overlap": mean_overlap, "num_proxy_rows": int(len(rows))}


def scale_bounds(bounds: dict, threshold_scale: float) -> dict:
    out = dict(bounds or {})
    for axis in ("x", "y", "z"):
        if axis not in out or out[axis] is None:
            continue
        lo, hi = [float(v) for v in out[axis]]
        center = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo) * float(threshold_scale)
        out[axis] = [center - half_width, center + half_width]
    return out


def scaled_safety_config(config: dict, threshold_scale: float) -> dict:
    cfg = json.loads(json.dumps(config or {}))
    safety = cfg.setdefault("safety", {})
    scalar_keys = {
        "collision_force_threshold",
        "excessive_force_threshold",
        "speed_threshold",
        "high_speed_threshold",
        "drop_velocity_threshold",
    }
    for key in scalar_keys:
        if key in safety and safety[key] is not None:
            safety[key] = float(safety[key]) * float(threshold_scale)
    if "boundary_bounds" in safety:
        safety["boundary_bounds"] = scale_bounds(safety["boundary_bounds"], threshold_scale)
    per_suite = safety.get("per_suite_overrides", {}) or {}
    for suite_cfg in per_suite.values():
        for key in scalar_keys:
            if key in suite_cfg and suite_cfg[key] is not None:
                suite_cfg[key] = float(suite_cfg[key]) * float(threshold_scale)
        if "boundary_bounds" in suite_cfg:
            suite_cfg["boundary_bounds"] = scale_bounds(suite_cfg["boundary_bounds"], threshold_scale)
    return cfg


def rollout_suite(tensor_path: Path) -> str | None:
    meta_path = tensor_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    suite = meta.get("suite", None)
    return None if suite is None else str(suite)


def relabel_episode(tensor_path: Path, threshold_scale: float, rollout_config: dict) -> np.ndarray | None:
    with safe_open(str(tensor_path), framework="np") as f:
        keys = set(f.keys())
        if not {"eef_positions", "contact_forces", "actions"}.issubset(keys):
            return None
        eef = f.get_tensor("eef_positions")
        contact = f.get_tensor("contact_forces")
        actions = f.get_tensor("actions")
    cfg = scaled_safety_config(rollout_config, threshold_scale=threshold_scale)
    return SafetyLabeler(cfg, suite=rollout_suite(tensor_path)).label_episode_arrays(
        {"eef_positions": eef, "contact_forces": contact, "actions": actions}
    )


def run_threshold_sensitivity(
    data_dir: Path,
    labels_full: pd.DataFrame,
    output_dir: Path,
    max_episodes: int,
    rollout_config: dict,
) -> dict[str, float]:
    ids = set(labels_full["episode_id"].astype(str))
    paths = [p for p in sorted(data_dir.rglob("rollout_*.safetensors")) if p.stem in ids]
    if max_episodes > 0:
        paths = paths[:max_episodes]
    rows = []
    category_vectors: dict[float, np.ndarray] = {}
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        episode_category = []
        active_steps = []
        for path in paths:
            labels = relabel_episode(path, threshold_scale=scale, rollout_config=rollout_config)
            if labels is None:
                continue
            episode_category.append(labels.any(axis=0).astype(np.int8))
            active_steps.append(labels.sum(axis=0).astype(np.int64))
        if not episode_category:
            continue
        ep_cat = np.stack(episode_category, axis=0)
        category_vectors[scale] = ep_cat.reshape(-1)
        step_counts = np.stack(active_steps, axis=0)
        for idx, category in enumerate(SAFETY_CATEGORIES):
            rows.append(
                {
                    "threshold_scale": scale,
                    "category": category,
                    "episode_positive_rate": float(ep_cat[:, idx].mean()),
                    "mean_active_steps": float(step_counts[:, idx].mean()),
                    "n_episodes": int(ep_cat.shape[0]),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty and 1.0 in category_vectors:
        base = category_vectors[1.0]
        consistency_rows = []
        for scale, vec in category_vectors.items():
            consistency_rows.append(
                {
                    "threshold_scale": scale,
                    "category_consistency_vs_1x": float((vec == base).mean()) if vec.size == base.size else float("nan"),
                }
            )
        consistency = pd.DataFrame(consistency_rows)
        df = df.merge(consistency, on="threshold_scale", how="left")
    df.to_csv(output_dir / "safety_threshold_sensitivity.csv", index=False)
    min_consistency = float(df["category_consistency_vs_1x"].min()) if "category_consistency_vs_1x" in df else float("nan")
    return {"min_threshold_category_consistency": min_consistency, "num_threshold_rows": int(len(df))}


def format_latex_table(csv_path: Path, tex_path: Path, columns: list[str]) -> None:
    df = pd.read_csv(csv_path)
    keep = [c for c in columns if c in df.columns]
    tex = df[keep].to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
    tex_path.write_text(tex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_full = pd.read_csv(args.labels_full_csv)
    top_df = pd.read_csv(args.top_features_csv)
    top_features = top_df["feature_idx"].astype(int).tolist()
    rollout_config = load_yaml(args.rollout_config) if str(args.rollout_config).strip() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        args.sae_checkpoint,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        device=device,
    )
    bundle = load_samples(
        data_dir=Path(args.data_dir),
        labels_full=labels_full,
        layer=args.layer,
        sae=sae,
        norm_factor=norm_factor,
        device=device,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        max_episodes=args.max_episodes,
    )

    diagnostics: dict[str, float] = {}
    diagnostics.update(
        run_interventions(
            bundle=bundle,
            top_features=top_features,
            sae=sae,
            norm_factor=norm_factor,
            seed=args.seed,
            top_k=args.top_k,
            random_trials=args.random_trials,
            output_dir=out_dir,
        )
    )
    diagnostics.update(run_proxy_validation(bundle=bundle, top_features=top_features, output_dir=out_dir))
    diagnostics.update(
        run_threshold_sensitivity(
            data_dir=Path(args.data_dir),
            labels_full=labels_full,
            output_dir=out_dir,
            max_episodes=args.max_episodes,
            rollout_config=rollout_config,
        )
    )
    diagnostics = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in diagnostics.items()}
    write_json(out_dir / "asap_experiment_summary.json", diagnostics)

    format_latex_table(
        out_dir / "feature_patching_and_ablation_summary.csv",
        out_dir / "table_feature_patching_and_ablation.tex",
        [
            "condition",
            "mean_progress_logit_delta",
            "mean_raw_action_readout_shift_l2",
            "mean_raw_activation_delta_l2",
        ],
    )
    format_latex_table(
        out_dir / "dose_response_top5_progress_features.csv",
        out_dir / "table_dose_response_top5.tex",
        ["multiplier", "mean_progress_logit_delta", "mean_raw_action_readout_shift_l2", "mean_raw_activation_delta_l2"],
    )
    format_latex_table(
        out_dir / "progress_proxy_validation.csv",
        out_dir / "table_progress_proxy_validation.tex",
        ["proxy", "spearman_rho", "p_value", "n_episodes"],
    )


if __name__ == "__main__":
    main()
