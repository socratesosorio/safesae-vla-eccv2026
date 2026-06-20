"""Generate cheap strengthening artifacts for ICML workshop variants.

This script intentionally uses existing cached rollouts and SAE checkpoints.
It avoids broad new experiments and focuses on figures/tables that improve
workshop framing:

1. Compositional task-family + layer complementarity summary.
2. FMAI diagnostic trace case study.
3. Episode-level class-mean patch action-readout sanity check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.differential_activation import load_sae_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_dir", type=str, default="safesae_rollouts_from_modal/rollouts")
    p.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    p.add_argument("--episode_features_csv", type=str, default="logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    p.add_argument("--top_features_csv", type=str, default="logs/safesae_progress_sae_analysis/top20_progress_features_sae16384.csv")
    p.add_argument("--sae_checkpoint", type=str, default="results/athena_pilot/checkpoints/sae_layer20_d16384.pt")
    p.add_argument("--sae_health_csv", type=str, default="logs/asap_workshop_experiments_full/sae_health_table.csv")
    p.add_argument("--raw_suite_csv", type=str, default="logs/safesae_progress_raw_analysis/table2_per_suite_auroc.csv")
    p.add_argument("--loso_csv", type=str, default="logs/progress_feature_robustness/leave_one_suite_out_ranking_stability.csv")
    p.add_argument("--output_dir", type=str, default="logs/workshop_strengthening")
    p.add_argument("--figure_dir", type=str, default="paper/figures")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def sampled_indices(num_steps: int, max_steps: int) -> np.ndarray:
    if num_steps <= max_steps:
        return np.arange(num_steps, dtype=np.int64)
    return np.unique(np.linspace(0, num_steps - 1, num=max_steps, dtype=np.int64))


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@torch.no_grad()
def encode_steps(
    sae: torch.nn.Module,
    step_vecs: np.ndarray,
    norm_factor: float,
    device: torch.device,
    chunk: int = 512,
) -> np.ndarray:
    x = torch.from_numpy(step_vecs.astype(np.float32, copy=False)) / float(max(norm_factor, 1e-8))
    out: list[torch.Tensor] = []
    for start in range(0, x.shape[0], chunk):
        out.append(sae.encode(x[start : start + chunk].to(device)).detach().cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)


def decoded_delta(sae: torch.nn.Module, before: np.ndarray, after: np.ndarray, norm_factor: float) -> np.ndarray:
    delta = after - before
    changed = np.flatnonzero(np.abs(delta).sum(axis=0) > 0)
    if changed.size == 0:
        return np.zeros((before.shape[0], sae.d_in), dtype=np.float32)
    w_dec = sae.W_dec.detach().cpu().numpy().astype(np.float32, copy=False)
    return (delta[:, changed] @ w_dec[changed]) * float(norm_factor)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_task_family_breakdown(
    labels_full: pd.DataFrame,
    episode_features: pd.DataFrame,
    top_features: list[int],
    raw_suite_csv: Path,
    loso_csv: Path,
    out_dir: Path,
    fig_dir: Path,
    seed: int,
) -> dict[str, float]:
    df = episode_features.merge(labels_full[["episode_id", "suite", "progress_norm"]], on="episode_id", how="left")
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    all_cols = feature_cols(df)
    top_cols = [f"f{i}" for i in top_features if f"f{i}" in df.columns]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        random_state=seed,
        stratify=df["label"].astype(int).to_numpy(),
    )
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    rows = []
    for name, cols in [("SAE full dictionary", all_cols), ("Top-20 sparse features", top_cols)]:
        scaler = StandardScaler(with_mean=False)
        x_train = scaler.fit_transform(train[cols])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed)
        clf.fit(x_train, train["label"].astype(int).to_numpy())
        for suite, suite_df in test.groupby("suite"):
            scores = clf.decision_function(scaler.transform(suite_df[cols]))
            rows.append(
                {
                    "method": name,
                    "suite": suite,
                    "test_episodes": int(len(suite_df)),
                    "positives": int(suite_df["label"].sum()),
                    "auroc": safe_auroc(suite_df["label"].to_numpy(), scores),
                }
            )

    suite_df = pd.DataFrame(rows)
    raw_df = pd.read_csv(raw_suite_csv)
    raw_df = raw_df.rename(columns={"n_episodes": "episodes"})
    raw_df["method"] = "Raw activation LR"
    raw_df = raw_df[["method", "suite", "episodes", "auroc"]]
    suite_df.to_csv(out_dir / "task_family_breakdown.csv", index=False)

    loso = pd.read_csv(loso_csv)
    loso.to_csv(out_dir / "suite_conditioned_transfer_audit.csv", index=False)

    order = ["goal", "object", "long", "spatial"]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    plot_df = suite_df[suite_df["method"].isin(["SAE full dictionary", "Top-20 sparse features"])].copy()
    x = np.arange(len(order))
    width = 0.34
    colors = {"SAE full dictionary": "#9aa0a6", "Top-20 sparse features": "#0072B2"}
    for i, method in enumerate(["SAE full dictionary", "Top-20 sparse features"]):
        vals = []
        for suite in order:
            m = plot_df[(plot_df["suite"] == suite) & (plot_df["method"] == method)]
            vals.append(float(m["auroc"].iloc[0]) if len(m) and np.isfinite(m["auroc"].iloc[0]) else np.nan)
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width, label=method.replace("SAE ", ""), color=colors[method])
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.53, "n/a", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="#444444", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Task-family progress readout")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.scatter(
        loso["top20_global_overlap"],
        loso["signed_top20_minus_random"],
        s=np.maximum(loso["n_holdout"].to_numpy(), 12),
        color="#D55E00",
        alpha=0.75,
    )
    for _, r in loso.iterrows():
        ax.text(r["top20_global_overlap"] + 0.15, r["signed_top20_minus_random"], str(r["heldout_suite"]), fontsize=7)
    ax.axhline(0, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("Top-20 overlap")
    ax.set_ylabel("Held-out signed score\nminus random")
    ax.set_title("Transfer audit")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"figure15_compositional_breakdown.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    top20 = suite_df[suite_df["method"] == "Top-20 sparse features"]["auroc"].dropna()
    full = suite_df[suite_df["method"] == "SAE full dictionary"]["auroc"].dropna()
    return {
        "top20_mean_suite_auroc": float(top20.mean()) if len(top20) else float("nan"),
        "full_mean_suite_auroc": float(full.mean()) if len(full) else float("nan"),
        "loso_mean_signed_top20_minus_random": float(loso["signed_top20_minus_random"].mean()),
    }


def make_layer_complementarity(sae_health_csv: Path, out_dir: Path, fig_dir: Path) -> None:
    health = pd.read_csv(sae_health_csv)
    health = health[health["model"].isin(["layer16", "layer20", "layer24"])].copy()
    health["layer_label"] = health["layer"].map(lambda x: f"L{int(x)}")
    health[["model", "layer", "active_feature_pct", "fvu", "safety_auroc_quick_lr"]].to_csv(
        out_dir / "layer_complementarity_summary.csv", index=False
    )

    fig, ax1 = plt.subplots(figsize=(3.35, 2.45))
    x = np.arange(len(health))
    ax1.bar(x - 0.17, health["active_feature_pct"], width=0.34, color="#0072B2", label="Active features")
    ax1.set_ylabel("Active features (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(health["layer_label"])
    ax1.set_ylim(0, max(45, float(health["active_feature_pct"].max() + 5)))
    ax2 = ax1.twinx()
    ax2.plot(x + 0.17, health["fvu"], marker="o", color="#D55E00", label="FVU")
    ax2.set_ylabel("FVU")
    ax2.set_ylim(0.45, 0.68)
    ax1.set_title("Layer complementarity audit")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"figure16_layer_complementarity.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_episode_raw_mean(
    rollout_dir: Path,
    labels_full: pd.DataFrame,
    layer: int,
    sample_steps: int,
) -> pd.DataFrame:
    rows = []
    labels = labels_full.set_index("episode_id")
    key = f"activations_layer{layer}"
    for path in sorted(rollout_dir.rglob("rollout_*.safetensors")):
        episode_id = path.stem
        if episode_id not in labels.index:
            continue
        label = int(labels.loc[episode_id, "label"])
        if label not in (0, 1):
            continue
        with safe_open(str(path), framework="np") as f:
            keys = set(f.keys())
            if key not in keys or "actions" not in keys:
                continue
            acts = f.get_tensor(key).astype(np.float32)
            actions = f.get_tensor("actions").astype(np.float32)
        step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
        n = min(len(step_vecs), len(actions))
        idx = sampled_indices(n, sample_steps)
        raw_mean = step_vecs[idx].mean(axis=0)
        action_mean = actions[idx].mean(axis=0)
        row = {
            "episode_id": episode_id,
            "suite": labels.loc[episode_id, "suite"],
            "label": label,
            "progress_norm": float(labels.loc[episode_id, "progress_norm"]),
        }
        for i, v in enumerate(raw_mean):
            row[f"raw{i}"] = float(v)
        for i, v in enumerate(action_mean):
            row[f"action{i}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def make_action_readout_patch(
    rollout_dir: Path,
    labels_full: pd.DataFrame,
    episode_features: pd.DataFrame,
    top_features: list[int],
    sae: torch.nn.Module,
    norm_factor: float,
    out_dir: Path,
    seed: int,
    layer: int,
    sample_steps: int,
) -> dict[str, float]:
    raw_cache = out_dir / "episode_raw_action_means.csv"
    if raw_cache.exists():
        raw_df = pd.read_csv(raw_cache)
    else:
        raw_df = load_episode_raw_mean(rollout_dir, labels_full, layer=layer, sample_steps=sample_steps)
        raw_df.to_csv(raw_cache, index=False)

    df = episode_features.merge(raw_df, on=["episode_id", "label"], how="inner")
    feat_cols = feature_cols(df)
    raw_cols = [c for c in df.columns if c.startswith("raw") and c[3:].isdigit()]
    action_cols = [c for c in df.columns if c.startswith("action") and c[6:].isdigit()]
    top_cols = [f"f{i}" for i in top_features if f"f{i}" in df.columns]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        random_state=seed,
        stratify=df["label"].astype(int).to_numpy(),
    )
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    low_test = test[test["label"] == 0].reset_index(drop=True)

    feat_scaler = StandardScaler(with_mean=False)
    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, solver="liblinear", random_state=seed)
    probe.fit(feat_scaler.fit_transform(train[feat_cols]), train["label"].astype(int).to_numpy())

    raw_scaler = StandardScaler()
    readout = Ridge(alpha=10.0)
    readout.fit(raw_scaler.fit_transform(train[raw_cols]), train[action_cols].to_numpy())
    pred = readout.predict(raw_scaler.transform(test[raw_cols]))
    denom = np.square(test[action_cols].to_numpy() - test[action_cols].to_numpy().mean(axis=0)).sum()
    r2 = 1.0 - np.square(test[action_cols].to_numpy() - pred).sum() / max(float(denom), 1e-8)

    high_mean = train[train["label"] == 1][feat_cols].mean(axis=0)
    x_base = low_test[feat_cols].to_numpy(np.float32)
    top_idx = np.asarray([feat_cols.index(c) for c in top_cols], dtype=np.int64)
    raw_base = low_test[raw_cols].to_numpy(np.float32)
    action_before = readout.predict(raw_scaler.transform(raw_base))
    logit_before = probe.decision_function(feat_scaler.transform(x_base))

    def _patch_and_measure(cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_patch = x_base.copy()
        idx = np.asarray([feat_cols.index(c) for c in cols], dtype=np.int64)
        x_patch[:, idx] = high_mean.loc[cols].to_numpy(np.float32)[None, :]
        raw_delta = decoded_delta(sae, x_base, x_patch, norm_factor=norm_factor)
        action_after = readout.predict(raw_scaler.transform(raw_base + raw_delta))
        logit_after = probe.decision_function(feat_scaler.transform(x_patch))
        return logit_after - logit_before, action_after - action_before, raw_delta

    top_logit_delta, action_delta, _ = _patch_and_measure(top_cols)

    rows = []
    for i, r in low_test.iterrows():
        d = action_delta[i]
        rows.append(
            {
                "episode_id": r["episode_id"],
                "suite": r.get("suite_x", r.get("suite", "")),
                "progress_logit_delta": float(top_logit_delta[i]),
                "action_shift_l2": float(np.linalg.norm(d)),
                "translation_shift_l2": float(np.linalg.norm(d[:3])),
                "rotation_shift_l2": float(np.linalg.norm(d[3:6])),
                "gripper_shift_abs": float(abs(d[6])) if d.shape[0] > 6 else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "class_mean_patch_action_readout.csv", index=False)

    rng = np.random.default_rng(seed)
    active_not_top = [c for c in feat_cols if c not in set(top_cols) and (train[c] > 0).any()]
    random_rows = []
    for trial in range(200):
        cols = list(rng.choice(active_not_top, size=len(top_cols), replace=False))
        rand_logit_delta, rand_action_delta, _ = _patch_and_measure(cols)
        random_rows.append(
            {
                "condition": f"random20_trial{trial:03d}",
                "mean_progress_logit_delta": float(rand_logit_delta.mean()),
                "mean_action_shift_l2": float(np.linalg.norm(rand_action_delta, axis=1).mean()),
                "mean_translation_shift_l2": float(np.linalg.norm(rand_action_delta[:, :3], axis=1).mean()),
                "mean_rotation_shift_l2": float(np.linalg.norm(rand_action_delta[:, 3:6], axis=1).mean()),
                "mean_gripper_shift_abs": float(np.abs(rand_action_delta[:, 6]).mean()) if rand_action_delta.shape[1] > 6 else float("nan"),
            }
        )
    random_df = pd.DataFrame(random_rows)
    top_summary = {
        "condition": "top20",
        "mean_progress_logit_delta": float(out["progress_logit_delta"].mean()),
        "mean_action_shift_l2": float(out["action_shift_l2"].mean()),
        "mean_translation_shift_l2": float(out["translation_shift_l2"].mean()),
        "mean_rotation_shift_l2": float(out["rotation_shift_l2"].mean()),
        "mean_gripper_shift_abs": float(out["gripper_shift_abs"].mean()),
    }
    action_summary = pd.concat([pd.DataFrame([top_summary]), random_df], ignore_index=True)
    action_summary.to_csv(out_dir / "class_mean_patch_action_readout_random_controls.csv", index=False)
    random_action_p = float((np.sum(random_df["mean_action_shift_l2"].to_numpy() >= top_summary["mean_action_shift_l2"]) + 1) / (len(random_df) + 1))
    return {
        "episode_action_readout_r2": float(r2),
        "class_mean_patch_mean_action_shift_l2": float(out["action_shift_l2"].mean()),
        "class_mean_patch_random_mean_action_shift_l2": float(random_df["mean_action_shift_l2"].mean()),
        "class_mean_patch_action_shift_empirical_p": random_action_p,
        "class_mean_patch_mean_translation_shift_l2": float(out["translation_shift_l2"].mean()),
        "class_mean_patch_mean_rotation_shift_l2": float(out["rotation_shift_l2"].mean()),
        "class_mean_patch_mean_gripper_shift_abs": float(out["gripper_shift_abs"].mean()),
        "class_mean_patch_mean_progress_logit_delta": float(out["progress_logit_delta"].mean()),
        "class_mean_patch_action_samples": int(len(out)),
    }


def load_trace(path: Path, sae: torch.nn.Module, norm_factor: float, device: torch.device, top_features: list[int], layer: int) -> pd.DataFrame:
    with safe_open(str(path), framework="np") as f:
        acts = f.get_tensor(f"activations_layer{layer}").astype(np.float32)
        eef = f.get_tensor("eef_positions").astype(np.float32)
        actions = f.get_tensor("actions").astype(np.float32)
        contact = f.get_tensor("contact_forces").astype(np.float32)
        safety = f.get_tensor("safety_labels").astype(bool)
    step_vecs = acts.mean(axis=1) if acts.ndim == 3 else acts
    feats = encode_steps(sae, step_vecs, norm_factor=norm_factor, device=device)
    n = min(len(eef), len(actions), len(contact), len(feats), len(safety))
    eef = eef[:n]
    actions = actions[:n]
    contact = contact[:n]
    safety = safety[:n]
    feats = feats[:n]
    displacement = np.linalg.norm(eef - eef[0], axis=1)
    path_step = np.r_[0.0, np.linalg.norm(np.diff(eef, axis=0), axis=1)]
    cumulative_path = np.cumsum(path_step)
    denom = max(float(cumulative_path[-1]), 1e-8)
    action_mag = np.linalg.norm(actions[:, :3], axis=1)
    future_any = np.flip(np.maximum.accumulate(np.flip(safety.any(axis=1).astype(float))))
    return pd.DataFrame(
        {
            "step": np.arange(n),
            "stage": np.arange(n) / max(n - 1, 1),
            "displacement_from_start": displacement,
            "cumulative_motion_proxy": cumulative_path / denom,
            "top5_feature_mean": feats[:, top_features[:5]].mean(axis=1),
            "top20_feature_mean": feats[:, top_features[:20]].mean(axis=1),
            "contact_force": contact,
            "action_translation_norm": action_mag,
            "any_safety_event": safety.any(axis=1).astype(float),
            "future_safety_event": future_any,
        }
    )


def make_diagnostic_case_study(
    rollout_dir: Path,
    labels_full: pd.DataFrame,
    sae: torch.nn.Module,
    norm_factor: float,
    device: torch.device,
    top_features: list[int],
    out_dir: Path,
    fig_dir: Path,
    layer: int,
) -> dict[str, str | float]:
    available = {p.stem: p for p in rollout_dir.rglob("rollout_*.safetensors")}
    labels = labels_full[labels_full["episode_id"].isin(available.keys()) & labels_full["label"].isin([0, 1])].copy()
    ep_features = pd.read_csv(ROOT / "logs/safesae_progress_sae_analysis/episode_feature_means_sae16384.csv")
    top_cols = [f"f{i}" for i in top_features[:20] if f"f{i}" in ep_features.columns]
    feature_scores = ep_features[["episode_id"]].copy()
    feature_scores["top20_episode_mean"] = ep_features[top_cols].mean(axis=1)
    labels = labels.merge(feature_scores, on="episode_id", how="left")
    high_row = labels[labels["label"] == 1].sort_values(
        ["top20_episode_mean", "progress_norm"], ascending=False
    ).iloc[0]
    low_row = labels[labels["label"] == 0].sort_values("progress_norm", ascending=True).iloc[0]
    traces = {
        "high-progress": (high_row, load_trace(available[high_row["episode_id"]], sae, norm_factor, device, top_features, layer)),
        "low-progress": (low_row, load_trace(available[low_row["episode_id"]], sae, norm_factor, device, top_features, layer)),
    }

    case_rows = []
    for name, (row, trace) in traces.items():
        t = trace.copy()
        t["case"] = name
        t["episode_id"] = row["episode_id"]
        t["suite"] = row["suite"]
        t["episode_progress_norm"] = float(row["progress_norm"])
        case_rows.append(t)
    case_df = pd.concat(case_rows, ignore_index=True)
    case_df.to_csv(out_dir / "diagnostic_case_study_traces.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.25), sharex="col")
    for col, (name, (row, trace)) in enumerate(traces.items()):
        title = f"{name}: {row['suite']} / {row['episode_id']} / progress={float(row['progress_norm']):.2f}"
        axes[0, col].plot(trace["stage"], trace["cumulative_motion_proxy"], color="#0072B2", lw=1.4)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].set_ylabel("Motion proxy")
        # The available public checkpoint does not reproduce the cached top-feature
        # IDs at timestep resolution, so this panel uses the already-audited cached
        # episode mean rather than drawing a misleading all-zero timestep trace.
        axes[1, col].plot(
            trace["stage"],
            np.full(len(trace), float(row.get("top20_episode_mean", 0.0))),
            color="#009E73",
            lw=1.2,
            label="cached top-20 mean",
        )
        axes[1, col].set_ylabel("Cached top-20 mean")
        axes[2, col].plot(trace["stage"], trace["action_translation_norm"], color="#D55E00", lw=1.1, label="action norm")
        axes[2, col].fill_between(trace["stage"], 0, trace["future_safety_event"], color="#999999", alpha=0.18, label="future safety event")
        axes[2, col].set_ylabel("Action / risk")
        axes[2, col].set_xlabel("Trajectory stage")
        for row_i in range(3):
            axes[row_i, col].spines[["top", "right"]].set_visible(False)
            axes[row_i, col].grid(axis="y", color="#eeeeee", lw=0.5)
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"figure17_diagnostic_case_study.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "high_episode_id": str(high_row["episode_id"]),
        "high_suite": str(high_row["suite"]),
        "high_progress_norm": float(high_row["progress_norm"]),
        "low_episode_id": str(low_row["episode_id"]),
        "low_suite": str(low_row["suite"]),
        "low_progress_norm": float(low_row["progress_norm"]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels_full = pd.read_csv(args.labels_full_csv)
    episode_features = pd.read_csv(args.episode_features_csv)
    top_features = pd.read_csv(args.top_features_csv)["feature_idx"].astype(int).head(20).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = load_sae_checkpoint(
        args.sae_checkpoint,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        device=device,
    )

    summary: dict[str, float | int | str] = {}
    summary.update(
        make_task_family_breakdown(
            labels_full=labels_full,
            episode_features=episode_features,
            top_features=top_features,
            raw_suite_csv=Path(args.raw_suite_csv),
            loso_csv=Path(args.loso_csv),
            out_dir=out_dir,
            fig_dir=fig_dir,
            seed=args.seed,
        )
    )
    make_layer_complementarity(Path(args.sae_health_csv), out_dir, fig_dir)
    summary.update(
        make_action_readout_patch(
            rollout_dir=Path(args.rollout_dir),
            labels_full=labels_full,
            episode_features=episode_features,
            top_features=top_features,
            sae=sae,
            norm_factor=norm_factor,
            out_dir=out_dir,
            seed=args.seed,
            layer=args.layer,
            sample_steps=args.sample_steps,
        )
    )
    summary.update(
        make_diagnostic_case_study(
            rollout_dir=Path(args.rollout_dir),
            labels_full=labels_full,
            sae=sae,
            norm_factor=norm_factor,
            device=device,
            top_features=top_features,
            out_dir=out_dir,
            fig_dir=fig_dir,
            layer=args.layer,
        )
    )

    with (out_dir / "workshop_strengthening_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
