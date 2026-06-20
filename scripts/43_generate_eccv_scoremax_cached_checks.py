"""Generate 24-hour rebuttal cached checks.

This script targets the non-closed-loop evidence that can still move reviewer
scores quickly:

- leave-one-suite-out progress generalization;
- early-prefix operating points at fixed false-positive rates;
- a compact robust-feature atlas.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_PATH = ROOT / "scripts" / "40_generate_eccv_temporal_phase_checks.py"


def _load_temporal_module():
    spec = importlib.util.spec_from_file_location("eccv_temporal_phase_checks", TEMPORAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TEMPORAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


temporal = _load_temporal_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="safesae_rollouts_from_modal/rollouts")
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--sae_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--raw_features_csv", type=str, default="logs/eccv_rebuttal_checks/episode_raw_layer20_means.csv")
    p.add_argument("--telemetry_csv", type=str, default="logs/eccv_rebuttal_checks/episode_telemetry_controls_and_semantic_audit.csv")
    p.add_argument("--robust_features_csv", type=str, default="logs/eccv_confound_controls_20260508-230421/episode_level_fdr.csv")
    p.add_argument("--phase_enrichment_csv", type=str, default="logs/eccv_temporal_phase_checks_20260509-013820/phase_feature_enrichment.csv")
    p.add_argument("--sae_checkpoint", type=str, default="results/athena_pilot/checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--output_dir", type=str, default="logs/eccv_scoremax_cached_checks_20260510")
    p.add_argument("--prefixes", type=str, default="0.10,0.25")
    p.add_argument("--max_timesteps_per_prefix", type=int, default=16)
    p.add_argument("--max_phase_steps_per_episode", type=int, default=64)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=12653)
    return p.parse_args()


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def global_quartile_labels(labels_full: pd.DataFrame) -> pd.DataFrame:
    df = labels_full[["episode_id", "suite", "progress_norm"]].copy()
    lo, hi = df["progress_norm"].quantile([0.25, 0.75])
    df = df[(df["progress_norm"] <= lo) | (df["progress_norm"] >= hi)].copy()
    df["label"] = (df["progress_norm"] >= hi).astype(int)
    return df.reset_index(drop=True)


def rank_train_top20(train_df: pd.DataFrame, cols: list[str]) -> list[str]:
    y = train_df["label"].to_numpy(dtype=int)
    x = train_df[cols].to_numpy(np.float32, copy=False)
    low = x[y == 0]
    high = x[y == 1]
    pooled = np.sqrt(0.5 * (low.var(axis=0) + high.var(axis=0))) + 1e-8
    effect = np.abs((high.mean(axis=0) - low.mean(axis=0)) / pooled)
    return [cols[int(i)] for i in np.argsort(-effect)[:20]]


def fit_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, seed: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "pca20_lr":
        n_comp = min(20, train_x.shape[1], len(train_y) - 1)
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, random_state=seed),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
        )
    else:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed),
        )
    model.fit(train_x, train_y)
    train_score = model.decision_function(train_x)
    test_score = model.decision_function(test_x)
    return train_score, test_score


def run_loso(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    labels = global_quartile_labels(pd.read_csv(args.labels_full_csv))
    sae = pd.read_csv(args.sae_features_csv).drop(columns=["label"], errors="ignore")
    raw = pd.read_csv(args.raw_features_csv)
    tel = pd.read_csv(args.telemetry_csv).drop(columns=["suite"], errors="ignore")
    df = labels.merge(sae, on="episode_id").merge(raw, on="episode_id").merge(tel, on="episode_id")
    fcols = feature_cols(df)
    rcols = raw_cols(df)
    motion_cols = [
        "eef_final_displacement",
        "eef_path_length",
        "eef_mean_velocity_proxy",
        "eef_max_velocity_proxy",
        "action_mean_norm",
        "action_max_norm",
        "action_translation_mean_norm",
        "contact_force_mean",
        "contact_force_max",
    ]
    rows = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for held in sorted(df["suite"].unique()):
            train = df[df["suite"] != held].reset_index(drop=True)
            test = df[df["suite"] == held].reset_index(drop=True)
            if test.empty or len(np.unique(train["label"])) < 2 or len(np.unique(test["label"])) < 2:
                continue
            y_train = train["label"].to_numpy(dtype=int)
            y_test = test["label"].to_numpy(dtype=int)
            nested = rank_train_top20(train, fcols)
            specs = [
                ("motion", motion_cols, "lr"),
                ("nested_top20_sae", nested, "lr"),
                ("full_sae", fcols, "lr"),
                ("raw_lr", rcols, "lr"),
                ("raw_pca20_lr", rcols, "pca20_lr"),
            ]
            for name, cols, method in specs:
                _, score = fit_score(
                    train[cols].to_numpy(np.float32, copy=False),
                    y_train,
                    test[cols].to_numpy(np.float32, copy=False),
                    seed=int(args.seed),
                    method=method,
                )
                rows.append(
                    {
                        "heldout_suite": held,
                        "method": name,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "test_positives": int(y_test.sum()),
                        "test_negatives": int((1 - y_test).sum()),
                        "auroc": safe_auroc(y_test, score),
                        "pr_auc": safe_ap(y_test, score),
                        "selected_features": ",".join(nested) if name == "nested_top20_sae" else "",
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "leave_one_suite_out.csv", index=False)
    return result


def op_metrics(y_true: np.ndarray, score: np.ndarray, train_y: np.ndarray, train_score: np.ndarray, fpr: float) -> dict[str, float]:
    neg_scores = train_score[train_y == 0]
    threshold = float(np.quantile(neg_scores, 1.0 - fpr))
    pred = score >= threshold
    pos = y_true == 1
    neg = ~pos
    tp = float((pred & pos).sum())
    fp = float((pred & neg).sum())
    return {
        "threshold": threshold,
        "tpr": tp / max(float(pos.sum()), 1.0),
        "fpr": fp / max(float(neg.sum()), 1.0),
        "precision": tp / max(float(pred.sum()), 1.0),
    }


def run_prefix_ops(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    labels = global_quartile_labels(pd.read_csv(args.labels_full_csv))
    y = labels["label"].to_numpy(dtype=int)
    top_features = pd.read_csv(args.robust_features_csv)["feature_idx"].astype(int).head(20).tolist()
    prefixes = temporal.parse_prefixes(args.prefixes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = temporal.load_sae_checkpoint(
        path=str(args.sae_checkpoint),
        d_in=int(args.d_in),
        d_sae=int(args.d_sae),
        k=int(args.k),
        device=device,
    )
    rollout_map = {p.stem: p for p in Path(args.rollout_dir).rglob("rollout_*.safetensors")}
    key = f"activations_layer{int(args.layer)}"
    packs = {prefix: {"y": [], "full": [], "motion": []} for prefix in prefixes}
    for _, row in labels.iterrows():
        path = rollout_map.get(str(row["episode_id"]))
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            if key not in f.keys():
                continue
            acts = f.get_tensor(key).astype(np.float32)
            eef = f.get_tensor("eef_positions").astype(np.float32)
            actions = f.get_tensor("actions").astype(np.float32)
            contact = f.get_tensor("contact_forces").astype(np.float32) if "contact_forces" in f.keys() else None
            safety = f.get_tensor("safety_labels").astype(bool) if "safety_labels" in f.keys() else None
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        t = int(step_vecs.shape[0])
        for prefix in prefixes:
            n = max(1, int(np.ceil(t * float(prefix))))
            idx = temporal.select_indices(n, int(args.max_timesteps_per_prefix))
            feats = temporal.encode_steps(sae, norm_factor, step_vecs[idx], device=device).mean(axis=0)
            packs[prefix]["y"].append(int(row["label"]))
            packs[prefix]["full"].append(feats)
            packs[prefix]["motion"].append(temporal.motion_features(eef=eef, actions=actions, contact=contact, safety=safety, n=n))
    rows = []
    rng = np.random.default_rng(int(args.seed))
    for prefix, pack in packs.items():
        yy = np.asarray(pack["y"], dtype=int)
        full = np.vstack(pack["full"]).astype(np.float32, copy=False)
        motion = np.vstack(pack["motion"]).astype(np.float32, copy=False)
        top = full[:, top_features]
        methods = {
            "robust20_sae": top,
            "motion": motion,
            "motion_plus_robust20_sae": np.concatenate([motion, top], axis=1),
        }
        # deterministic fold assignment with the temporal helper
        for method, x in methods.items():
            scores = np.full(len(yy), np.nan, dtype=np.float64)
            train_scores = []
            fold_rows = []
            for fold, (train_idx, test_idx) in enumerate(temporal.fold_iterator(yy, folds=5, seed=int(args.seed))):
                train_score, test_score = fit_score(x[train_idx], yy[train_idx], x[test_idx], seed=int(args.seed), method="lr")
                scores[test_idx] = test_score
                train_scores.append((yy[train_idx], train_score, test_idx, test_score))
                for fpr_target in [0.10, 0.20]:
                    metrics = op_metrics(yy[test_idx], test_score, yy[train_idx], train_score, fpr_target)
                    fold_rows.append({"fpr_target": fpr_target, **metrics})
            for fpr_target in [0.10, 0.20]:
                sub = [r for r in fold_rows if r["fpr_target"] == fpr_target]
                rows.append(
                    {
                        "prefix_fraction": float(prefix),
                        "method": method,
                        "n": int(len(yy)),
                        "auroc": safe_auroc(yy, scores),
                        "pr_auc": safe_ap(yy, scores),
                        "fpr_target": float(fpr_target),
                        "mean_realized_fpr": float(np.mean([r["fpr"] for r in sub])),
                        "mean_tpr": float(np.mean([r["tpr"] for r in sub])),
                        "mean_precision": float(np.mean([r["precision"] for r in sub])),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "early_prefix_operating_points.csv", index=False)
    return result


def corr_safe(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.unique(a[finite]).size < 2 or np.unique(b[finite]).size < 2:
        return float("nan")
    if kind == "spearman":
        return float(spearmanr(a[finite], b[finite]).correlation)
    return float(pearsonr(a[finite], b[finite])[0])


def run_feature_atlas(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    labels = global_quartile_labels(pd.read_csv(args.labels_full_csv))
    sae = pd.read_csv(args.sae_features_csv).drop(columns=["label"], errors="ignore")
    tel = pd.read_csv(args.telemetry_csv).drop(columns=["suite"], errors="ignore")
    df = labels.merge(sae, on="episode_id").merge(tel, on="episode_id")
    robust = pd.read_csv(args.robust_features_csv)["feature_idx"].astype(int).head(8).tolist()
    phase = pd.read_csv(args.phase_enrichment_csv) if Path(args.phase_enrichment_csv).exists() else pd.DataFrame()
    telemetry_cols = [
        "eef_final_displacement",
        "eef_path_length",
        "eef_mean_velocity_proxy",
        "eef_max_velocity_proxy",
        "action_mean_norm",
        "action_max_norm",
        "action_translation_mean_norm",
        "contact_force_mean",
        "contact_force_max",
    ]
    rows = []
    for feat in robust:
        col = f"f{feat}"
        if col not in df.columns:
            continue
        low = df[df["label"] == 0][col].to_numpy(float)
        high = df[df["label"] == 1][col].to_numpy(float)
        corrs = []
        for tcol in telemetry_cols:
            corrs.append((tcol, abs(corr_safe(df[col].to_numpy(float), df[tcol].to_numpy(float), "spearman"))))
        best_tel, best_corr = sorted(corrs, key=lambda x: -np.nan_to_num(x[1], nan=-1.0))[0]
        phase_name = ""
        phase_delta = float("nan")
        if not phase.empty:
            sub = phase[phase["feature_idx"] == feat].copy()
            if not sub.empty:
                sub["abs_delta"] = sub["delta_phase_minus_other"].abs()
                row = sub.sort_values("abs_delta", ascending=False).iloc[0]
                phase_name = str(row["phase"])
                phase_delta = float(row["delta_phase_minus_other"])
        rows.append(
            {
                "feature_idx": int(feat),
                "mean_low": float(np.mean(low)),
                "mean_high": float(np.mean(high)),
                "activation_rate_low": float((low > 1e-8).mean()),
                "activation_rate_high": float((high > 1e-8).mean()),
                "direction": "higher_in_high_progress" if float(np.mean(high)) >= float(np.mean(low)) else "higher_in_low_progress",
                "strongest_telemetry_correlation": best_tel,
                "abs_spearman": float(best_corr),
                "peak_phase_proxy": phase_name,
                "phase_delta": phase_delta,
                "cautious_label": "progress-associated sparse direction",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "robust_feature_mini_atlas.csv", index=False)
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loso = run_loso(args, out_dir)
    ops = run_prefix_ops(args, out_dir)
    atlas = run_feature_atlas(args, out_dir)
    summary = {
        "leave_one_suite_out_best": loso.sort_values(["heldout_suite", "auroc"], ascending=[True, False]).groupby("heldout_suite").head(3).to_dict(orient="records"),
        "operating_points_best": ops.sort_values(["prefix_fraction", "fpr_target", "mean_tpr"], ascending=[True, True, False]).groupby(["prefix_fraction", "fpr_target"]).head(3).to_dict(orient="records"),
        "feature_atlas": atlas.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
