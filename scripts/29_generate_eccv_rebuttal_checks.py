"""Generate compact ECCV rebuttal checks from cached progress artifacts.

The rebuttal needs direct answers to reviewer requests, not a new experiment
suite. This script keeps to cached rollout activations and episode-level SAE
features and writes a small, auditable set of numbers:

- raw activation LR/MLP progress baselines;
- SAE LR/top-20 baselines under the same folds;
- bootstrap confidence intervals for headline AUROCs;
- split robustness for quartile, tertile, and median progress labels;
- per-suite AUROC CIs for the data-size concern;
- a semantic/proxy availability audit for success and object-state labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="safesae_rollouts_from_modal/rollouts")
    p.add_argument("--episode_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--sae_checkpoint", type=str, default="results/athena_pilot/checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_rebuttal_checks")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def ci_from_values(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.nanpercentile(values, [2.5, 97.5]))


def bootstrap_auroc_ci(y_true: np.ndarray, scores: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], scores[idx]))
    return ci_from_values(np.asarray(vals, dtype=float))


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def load_raw_episode_means(rollout_dir: Path, episode_ids: set[str], *, layer: int, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    key = f"activations_layer{layer}"
    for path in sorted(rollout_dir.rglob("rollout_*.safetensors")):
        ep = path.stem
        if ep not in episode_ids:
            continue
        with safe_open(str(path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)
        vec = acts.reshape(-1, acts.shape[-1]).mean(axis=0)
        row = {"episode_id": ep}
        row.update({f"r{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def load_telemetry_episode_features(rollout_dir: Path, episode_ids: set[str], *, cache_path: Path) -> pd.DataFrame:
    """Load lightweight non-representation controls and audit label availability."""
    if cache_path.exists():
        return pd.read_csv(cache_path)

    semantic_tensor_keys = {
        "object_states",
        "object_positions",
        "object_state",
        "goal_position",
        "target_position",
        "target_object_position",
        "task_completion",
        "completion_fraction",
        "reward",
        "rewards",
        "rgb",
        "image",
        "images",
        "frames",
    }
    semantic_json_keys = {
        "object_states",
        "object_positions",
        "object_state",
        "goal_position",
        "target_position",
        "target_object_position",
        "goal_xyz",
        "target_xyz",
        "task_completion",
        "completion_fraction",
        "progress",
        "success_fraction",
        "reward",
        "rewards",
        "episode_success",
        "success",
    }

    rows = []
    for path in sorted(rollout_dir.rglob("rollout_*.safetensors")):
        ep = path.stem
        if ep not in episode_ids:
            continue
        meta_path = path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            eef = f.get_tensor("eef_positions").astype(np.float32) if "eef_positions" in keys else None
            actions = f.get_tensor("actions").astype(np.float32) if "actions" in keys else None
            forces = f.get_tensor("contact_forces").astype(np.float32) if "contact_forces" in keys else None
            success_arr = f.get_tensor("episode_success") if "episode_success" in keys else np.asarray([False])

        if eef is None or actions is None:
            continue
        eef = eef.reshape(-1, 3)
        actions = actions.reshape(actions.shape[0], -1)
        eef_deltas = np.linalg.norm(np.diff(eef, axis=0), axis=1) if len(eef) > 1 else np.zeros((0,), dtype=np.float32)
        action_norms = np.linalg.norm(actions, axis=1)
        trans_norms = np.linalg.norm(actions[:, :3], axis=1) if actions.shape[1] >= 3 else action_norms
        row = {
            "episode_id": ep,
            "suite": str(meta.get("suite", "unknown")),
            "episode_success": int(bool(np.asarray(success_arr).astype(bool).any()) or bool(meta.get("episode_success", False))),
            "eef_final_displacement": float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) > 1 else 0.0,
            "eef_path_length": float(eef_deltas.sum()),
            "eef_mean_velocity_proxy": float(eef_deltas.mean()) if len(eef_deltas) else 0.0,
            "eef_max_velocity_proxy": float(eef_deltas.max()) if len(eef_deltas) else 0.0,
            "action_mean_norm": float(action_norms.mean()),
            "action_max_norm": float(action_norms.max()),
            "action_translation_mean_norm": float(trans_norms.mean()),
            "contact_force_mean": float(np.asarray(forces).mean()) if forces is not None else 0.0,
            "contact_force_max": float(np.asarray(forces).max()) if forces is not None else 0.0,
            "has_semantic_tensor_key": int(bool(keys & semantic_tensor_keys)),
            "has_semantic_json_key": int(bool(set(meta.keys()) & semantic_json_keys)),
            "available_tensor_keys": " ".join(sorted(keys)),
            "available_json_keys": " ".join(sorted(meta.keys())),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


@torch.no_grad()
def load_sae_episode_means_all(
    rollout_dir: Path,
    episode_ids: set[str],
    *,
    layer: int,
    checkpoint: Path,
    d_in: int,
    d_sae: int,
    k: int,
    cache_path: Path,
    max_timesteps: int = 8,
) -> pd.DataFrame:
    """Encode all episodes for split sensitivity.

    This is intentionally cached separately from the submitted quartile feature
    table, which only contains the labeled extremes.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        path=str(checkpoint),
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        device=device,
    )
    key = f"activations_layer{layer}"
    rows = []
    for path in sorted(rollout_dir.rglob("rollout_*.safetensors")):
        ep = path.stem
        if ep not in episode_ids:
            continue
        with safe_open(str(path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        if step_vecs.shape[0] > max_timesteps:
            idx = np.unique(np.linspace(0, step_vecs.shape[0] - 1, max_timesteps, dtype=np.int64))
            step_vecs = step_vecs[idx]
        x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)).to(device) / float(max(norm_factor, 1e-8))
        feats = sae.encode(x).detach().cpu().numpy().astype(np.float32, copy=False)
        vec = feats.mean(axis=0)
        row = {"episode_id": ep}
        row.update({f"f{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def make_labels(labels_full: pd.DataFrame, scheme: str) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    if scheme == "quartile":
        lo, hi = df["progress_norm"].quantile([0.25, 0.75])
        df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
        df["label"] = (df["progress_norm"] >= hi).astype(int)
    elif scheme == "tertile":
        lo, hi = df["progress_norm"].quantile([1 / 3, 2 / 3])
        df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
        df["label"] = (df["progress_norm"] >= hi).astype(int)
    elif scheme == "median":
        med = df["progress_norm"].median()
        df["label"] = (df["progress_norm"] >= med).astype(int)
    else:
        raise ValueError(f"unknown label scheme: {scheme}")
    return df.reset_index(drop=True)


def fit_predict_cv(x: np.ndarray, y: np.ndarray, *, model_name: str, seed: int, folds: int) -> dict:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "fold_aurocs": [], "scores": []}
    n_splits = min(int(folds), int(np.bincount(y).min()))
    if n_splits < 2:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y)
        splits = [(train_idx, test_idx)]
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)

    all_y = []
    all_scores = []
    fold_aurocs = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in splits:
            if model_name == "lr":
                model = make_pipeline(
                    StandardScaler(with_mean=False),
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
                )
            elif model_name == "raw_mlp":
                model = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                )
            elif model_name == "pca_lr":
                n_comp = min(64, x.shape[1], len(train_idx) - 1)
                model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=n_comp, random_state=seed),
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
                )
            elif model_name == "pca20_lr":
                n_comp = min(20, x.shape[1], len(train_idx) - 1)
                model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=n_comp, random_state=seed),
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
                )
            elif model_name == "rp20_lr":
                n_comp = min(20, x.shape[1])
                model = make_pipeline(
                    StandardScaler(),
                    GaussianRandomProjection(n_components=n_comp, random_state=seed),
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
                )
            else:
                raise ValueError(model_name)
            model.fit(x[train_idx], y[train_idx])
            if hasattr(model[-1], "decision_function"):
                scores = model.decision_function(x[test_idx])
            else:
                scores = model.predict_proba(x[test_idx])[:, 1]
            fold_aurocs.append(safe_auroc(y[test_idx], scores))
            all_y.append(y[test_idx])
            all_scores.append(scores)
    yy = np.concatenate(all_y)
    ss = np.concatenate(all_scores)
    pred = (ss >= 0).astype(int) if np.nanmin(ss) < 0 else (ss >= 0.5).astype(int)
    return {
        "auroc": safe_auroc(yy, ss),
        "pr_auc": float(average_precision_score(yy, ss)) if len(np.unique(yy)) > 1 else float("nan"),
        "f1": float(f1_score(yy, pred, zero_division=0)),
        "precision": float(precision_score(yy, pred, zero_division=0)),
        "recall": float(recall_score(yy, pred, zero_division=0)),
        "fold_aurocs": [float(v) for v in fold_aurocs],
        "scores": ss.tolist(),
        "y": yy.tolist(),
    }


def evaluate_scheme(
    *,
    scheme: str,
    labels_full: pd.DataFrame,
    sae_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    top_features: list[int],
    seed: int,
    folds: int,
    bootstrap: int,
) -> tuple[list[dict], dict]:
    labels = make_labels(labels_full, scheme)
    sae_use = sae_df.drop(columns=[c for c in ("label", "suite", "progress_norm") if c in sae_df.columns])
    tel_use = telemetry_df.drop(columns=[c for c in ("suite",) if c in telemetry_df.columns])
    df = labels.merge(sae_use, on="episode_id", how="inner").merge(raw_df, on="episode_id", how="inner").merge(tel_use, on="episode_id", how="inner")
    feat = feature_cols(df)
    active = [c for c in feat if (df[c] > 0).any()]
    top = [f"f{i}" for i in top_features if f"f{i}" in df.columns]
    rng = np.random.default_rng(seed + 700 + len(scheme))
    active_not_top = [c for c in active if c not in set(top)]
    random20 = list(rng.choice(active_not_top if len(active_not_top) >= 20 else active, size=20, replace=False))
    raw = [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]
    motion = [
        "eef_mean_velocity_proxy",
        "eef_max_velocity_proxy",
        "eef_path_length",
        "action_mean_norm",
        "action_max_norm",
        "action_translation_mean_norm",
        "contact_force_mean",
        "contact_force_max",
    ]
    y = df["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed + len(scheme))

    specs = [
        ("SAE LR", df[active].to_numpy(dtype=np.float32), "lr"),
        ("Top-20 SAE LR", df[top].to_numpy(dtype=np.float32), "lr"),
        ("Raw activation LR", df[raw].to_numpy(dtype=np.float32), "lr"),
        ("Raw activation MLP", df[raw].to_numpy(dtype=np.float32), "raw_mlp"),
        ("Raw PCA-64 LR", df[raw].to_numpy(dtype=np.float32), "pca_lr"),
        ("Raw PCA-20 LR", df[raw].to_numpy(dtype=np.float32), "pca20_lr"),
        ("Raw RP-20 LR", df[raw].to_numpy(dtype=np.float32), "rp20_lr"),
        ("Motion telemetry LR", df[motion].to_numpy(dtype=np.float32), "lr"),
        ("Random-20 SAE LR", df[random20].to_numpy(dtype=np.float32), "lr"),
    ]

    rows = []
    details = {"n": int(len(df)), "positives": int(y.sum()), "negatives": int((1 - y).sum())}
    for method, x, model in specs:
        res = fit_predict_cv(x, y, model_name=model, seed=seed, folds=folds)
        ci = bootstrap_auroc_ci(np.asarray(res["y"], dtype=int), np.asarray(res["scores"], dtype=float), rng=rng, n_boot=bootstrap)
        row = {
            "scheme": scheme,
            "method": method,
            "n": int(len(df)),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "auroc": float(res["auroc"]),
            "auroc_ci95_low": ci[0],
            "auroc_ci95_high": ci[1],
            "pr_auc": float(res["pr_auc"]),
            "f1": float(res["f1"]),
            "precision": float(res["precision"]),
            "recall": float(res["recall"]),
            "fold_auroc_mean": float(np.nanmean(res["fold_aurocs"])),
            "fold_auroc_std": float(np.nanstd(res["fold_aurocs"])),
        }
        rows.append(row)
    return rows, details


def per_suite_uncertainty(
    labels_full: pd.DataFrame,
    sae_df: pd.DataFrame,
    top_features: list[int],
    *,
    seed: int,
    bootstrap: int,
) -> list[dict]:
    labels = make_labels(labels_full, "quartile")
    sae_use = sae_df.drop(columns=[c for c in ("label", "suite", "progress_norm") if c in sae_df.columns])
    df = labels.merge(sae_use, on="episode_id", how="inner")
    top = [f"f{i}" for i in top_features if f"f{i}" in df.columns]
    rows = []
    rng = np.random.default_rng(seed + 100)
    for suite, part in sorted(df.groupby("suite")):
        y = part["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            rows.append({"suite": suite, "n": int(len(part)), "positives": int(y.sum()), "negatives": int((1 - y).sum()), "top20_auroc": float("nan"), "top20_auroc_ci95_low": float("nan"), "top20_auroc_ci95_high": float("nan")})
            continue
        res = fit_predict_cv(part[top].to_numpy(dtype=np.float32), y, model_name="lr", seed=seed, folds=5)
        ci = bootstrap_auroc_ci(np.asarray(res["y"], dtype=int), np.asarray(res["scores"], dtype=float), rng=rng, n_boot=bootstrap)
        rows.append(
            {
                "suite": suite,
                "n": int(len(part)),
                "positives": int(y.sum()),
                "negatives": int((1 - y).sum()),
                "top20_auroc": float(res["auroc"]),
                "top20_auroc_ci95_low": ci[0],
                "top20_auroc_ci95_high": ci[1],
            }
        )
    return rows


def semantic_proxy_audit(labels_full: pd.DataFrame, telemetry_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = labels_full.merge(telemetry_df, on="episode_id", how="inner", suffixes=("", "_telemetry"))
    rows = []
    for proxy in [
        "episode_success",
        "eef_final_displacement",
        "eef_path_length",
        "eef_mean_velocity_proxy",
        "action_mean_norm",
        "contact_force_mean",
    ]:
        if proxy not in df.columns:
            continue
        x = pd.to_numeric(df["progress_norm"], errors="coerce")
        y = pd.to_numeric(df[proxy], errors="coerce")
        mask = x.notna() & y.notna()
        rho = float(x[mask].corr(y[mask], method="spearman")) if int(mask.sum()) >= 3 and y[mask].nunique() > 1 else float("nan")
        row = {
            "proxy": proxy,
            "n_episodes": int(mask.sum()),
            "spearman_rho_with_progress_norm": rho,
            "unique_values": int(y[mask].nunique()) if int(mask.sum()) else 0,
        }
        if proxy == "episode_success" and y[mask].nunique() == 2:
            row["progress_predicts_success_auroc"] = safe_auroc(y[mask].to_numpy(dtype=int), x[mask].to_numpy(dtype=float))
        else:
            row["progress_predicts_success_auroc"] = float("nan")
        rows.append(row)
    audit_df = pd.DataFrame(rows)
    semantic_tensor_count = int(pd.to_numeric(df.get("has_semantic_tensor_key", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    semantic_json_count = int(pd.to_numeric(df.get("has_semantic_json_key", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    object_or_goal_json_count = int(
        df["available_json_keys"].fillna("").str.contains(
            r"object_states|object_positions|object_state|goal_position|target_position|target_object_position|goal_xyz|target_xyz|task_completion|completion_fraction|success_fraction|reward",
            regex=True,
        ).sum()
    )
    object_or_goal_tensor_count = int(
        df["available_tensor_keys"].fillna("").str.contains(
            r"object_states|object_positions|object_state|goal_position|target_position|target_object_position|task_completion|completion_fraction|reward|rgb|image|images|frames",
            regex=True,
        ).sum()
    )
    success_pos = int(pd.to_numeric(df["episode_success"], errors="coerce").fillna(0).sum())
    summary = {
        "episodes_audited": int(len(df)),
        "episode_success_positive_count": success_pos,
        "episode_success_rate": float(success_pos / max(len(df), 1)),
        "semantic_tensor_key_episode_count": semantic_tensor_count,
        "semantic_json_key_episode_count": semantic_json_count,
        "object_or_goal_state_tensor_episode_count": object_or_goal_tensor_count,
        "object_or_goal_state_json_episode_count": object_or_goal_json_count,
        "metric_source_counts": {str(k): int(v) for k, v in labels_full["metric_source"].value_counts().items()} if "metric_source" in labels_full.columns else {},
    }
    return audit_df, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_full = pd.read_csv(args.labels_full_csv)
    sae_df = pd.read_csv(args.episode_features_csv)
    top_features = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(20).tolist()
    all_eps = set(labels_full["episode_id"].astype(str))
    raw_df = load_raw_episode_means(
        Path(args.rollout_dir),
        all_eps,
        layer=args.layer,
        cache_path=out_dir / f"episode_raw_layer{args.layer}_means.csv",
    )
    telemetry_df = load_telemetry_episode_features(
        Path(args.rollout_dir),
        all_eps,
        cache_path=out_dir / "episode_telemetry_controls_and_semantic_audit.csv",
    )
    sae_all_df = load_sae_episode_means_all(
        Path(args.rollout_dir),
        all_eps,
        layer=args.layer,
        checkpoint=Path(args.sae_checkpoint),
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        cache_path=out_dir / f"episode_sae_layer{args.layer}_d{args.d_sae}_means_all.csv",
    )

    rows: list[dict] = []
    details: dict[str, dict] = {}
    # Quartile uses the submitted cached SAE table. Tertile/median need the
    # all-episode SAE cache because the submitted table intentionally omitted
    # middle-quartile episodes.
    for scheme, scheme_sae_df in (
        ("quartile", sae_df),
        ("tertile", sae_all_df),
        ("median", sae_all_df),
    ):
        scheme_rows, scheme_details = evaluate_scheme(
            scheme=scheme,
            labels_full=labels_full,
            sae_df=scheme_sae_df,
            raw_df=raw_df,
            telemetry_df=telemetry_df,
            top_features=top_features,
            seed=args.seed,
            folds=args.folds,
            bootstrap=args.bootstrap,
        )
        rows.extend(scheme_rows)
        details[scheme] = scheme_details
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "rebuttal_progress_baselines_and_splits.csv", index=False)

    suite_df = pd.DataFrame(
        per_suite_uncertainty(labels_full, sae_df, top_features, seed=args.seed, bootstrap=args.bootstrap)
    )
    suite_df.to_csv(out_dir / "rebuttal_per_suite_uncertainty.csv", index=False)

    semantic_df, semantic_summary = semantic_proxy_audit(labels_full, telemetry_df)
    semantic_df.to_csv(out_dir / "rebuttal_semantic_proxy_audit.csv", index=False)

    quartile = metrics_df[metrics_df["scheme"] == "quartile"].set_index("method")
    summary = {
        "label_scheme_details": details,
        "quartile_sae_lr_auroc": float(quartile.loc["SAE LR", "auroc"]),
        "quartile_top20_auroc": float(quartile.loc["Top-20 SAE LR", "auroc"]),
        "quartile_raw_lr_auroc": float(quartile.loc["Raw activation LR", "auroc"]),
        "quartile_raw_mlp_auroc": float(quartile.loc["Raw activation MLP", "auroc"]),
        "quartile_raw_pca64_lr_auroc": float(quartile.loc["Raw PCA-64 LR", "auroc"]),
        "quartile_raw_pca20_lr_auroc": float(quartile.loc["Raw PCA-20 LR", "auroc"]),
        "quartile_raw_rp20_lr_auroc": float(quartile.loc["Raw RP-20 LR", "auroc"]),
        "quartile_motion_telemetry_lr_auroc": float(quartile.loc["Motion telemetry LR", "auroc"]),
        "quartile_random20_sae_lr_auroc": float(quartile.loc["Random-20 SAE LR", "auroc"]),
        "tertile_sae_lr_auroc": float(metrics_df[(metrics_df["scheme"] == "tertile") & (metrics_df["method"] == "SAE LR")]["auroc"].iloc[0]),
        "median_top20_auroc": float(metrics_df[(metrics_df["scheme"] == "median") & (metrics_df["method"] == "Top-20 SAE LR")]["auroc"].iloc[0]),
        "median_sae_lr_auroc": float(metrics_df[(metrics_df["scheme"] == "median") & (metrics_df["method"] == "SAE LR")]["auroc"].iloc[0]),
        "tertile_top20_auroc": float(metrics_df[(metrics_df["scheme"] == "tertile") & (metrics_df["method"] == "Top-20 SAE LR")]["auroc"].iloc[0]),
        "spatial_suite_n": int(suite_df[suite_df["suite"] == "spatial"]["n"].iloc[0]) if "spatial" in set(suite_df["suite"]) else 0,
        "spatial_top20_auroc": float(suite_df[suite_df["suite"] == "spatial"]["top20_auroc"].iloc[0]) if "spatial" in set(suite_df["suite"]) else float("nan"),
        "semantic_proxy_audit": semantic_summary,
    }
    write_json(out_dir / "rebuttal_checks_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
