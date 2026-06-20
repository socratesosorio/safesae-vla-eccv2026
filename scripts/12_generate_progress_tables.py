"""Regenerate progress-paper LaTeX tables from local progress artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate progress-paper tables from logs")
    parser.add_argument("--labels_csv", type=str, default="logs/safesae_progress_labels/progress_labels.csv")
    parser.add_argument("--labels_full_csv", type=str, default="logs/safesae_progress_labels/progress_labels_full.csv")
    parser.add_argument(
        "--monitor_metrics_json",
        type=str,
        default="logs/safesae_progress_sae_analysis/monitor_metrics_sae16384.json",
    )
    parser.add_argument(
        "--per_suite_csv",
        type=str,
        default="logs/safesae_progress_sae_analysis/per_suite_auroc_sae16384_split_by_suite.csv",
    )
    parser.add_argument(
        "--rollout_dir",
        type=str,
        default="safesae_rollouts_from_modal/rollouts",
        help="Rollout root used for timestep counts + motion-control baselines",
    )
    parser.add_argument("--output_dir", type=str, default="paper/tables")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion_lr_c", type=float, default=0.1)
    parser.add_argument("--default_episode_steps", type=int, default=600)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def format_float(x: float | None) -> str:
    if x is None or not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.3f}"


def bold_if(s: str, cond: bool) -> str:
    if s == "--":
        return s
    return f"\\textbf{{{s}}}" if cond else s


def resolve_metric_row(metrics: dict[str, Any], candidate_keys: list[str]) -> dict[str, float]:
    for key in candidate_keys:
        row = metrics.get(key)
        if isinstance(row, dict):
            return {
                "auroc": float(row.get("auroc", np.nan)),
                "f1": float(row.get("f1", np.nan)),
                "precision": float(row.get("precision", np.nan)),
                "recall": float(row.get("recall", np.nan)),
                "pr_auc": float(row.get("pr_auc", np.nan)),
            }
    return {"auroc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan, "pr_auc": np.nan}


def write_table_main_results(monitor_metrics_path: Path, out_path: Path) -> None:
    metrics = read_json(monitor_metrics_path)
    rows = [
        ("SAE Feature LR (16384-d)", resolve_metric_row(metrics, ["raw_activation_lr", "sae_lr_16384d", "sae_feature_lr"])),
        ("Top-20 Feature LR", resolve_metric_row(metrics, ["top20_feature_lr", "top20_lr"])),
        ("Random", resolve_metric_row(metrics, ["random"])),
    ]

    cols = ["auroc", "f1", "precision", "recall", "pr_auc"]
    best = {}
    for col in cols:
        vals = np.array([r[1][col] for r in rows], dtype=np.float64)
        if np.all(~np.isfinite(vals)):
            best[col] = np.nan
        else:
            best[col] = float(np.nanmax(vals))

    lines = [
        "\\begin{tabular}{lccccc}\\toprule",
        "Method & AUROC$\\uparrow$ & F1$\\uparrow$ & Precision$\\uparrow$ & Recall$\\uparrow$ & PR-AUC$\\uparrow$ \\\\ \\midrule",
    ]
    for method, row in rows:
        vals = []
        for col in cols:
            sval = format_float(row[col])
            is_best = np.isfinite(best[col]) and np.isfinite(row[col]) and abs(float(row[col]) - best[col]) < 1e-12
            vals.append(bold_if(sval, is_best))
        lines.append(f"{method} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write_text(out_path, "\n".join(lines) + "\n")


def write_table_category_auroc(per_suite_path: Path, out_path: Path) -> None:
    df = pd.read_csv(per_suite_path)
    if "suite" not in df.columns or "episodes" not in df.columns or "auroc" not in df.columns:
        raise ValueError(f"Unexpected per-suite schema in {per_suite_path}")

    preferred_order = ["goal", "object", "long", "spatial"]
    order_rank = {name: i for i, name in enumerate(preferred_order)}
    df = df.copy()
    df["_rank"] = df["suite"].map(lambda x: order_rank.get(str(x), 999))
    df = df.sort_values(["_rank", "suite"]).drop(columns=["_rank"])

    interpretation = {
        "goal": "Near-perfect progress separation",
        "object": "Strong and stable separation",
        "long": "Moderate, above chance",
        "spatial": "Positive but data-limited regime",
    }

    best_auroc = float(df["auroc"].max()) if not df.empty else np.nan
    lines = [
        "\\begin{tabular}{lccc}\\toprule",
        "Task Suite & Progress AUROC & Episodes & Interpretation \\\\ \\midrule",
    ]
    for _, row in df.iterrows():
        suite = str(row["suite"])
        auroc = float(row["auroc"])
        auroc_s = format_float(auroc)
        if np.isfinite(best_auroc) and abs(auroc - best_auroc) < 1e-12:
            auroc_s = f"\\textbf{{{auroc_s}}}"
        eps = int(row["episodes"])
        interp = interpretation.get(suite, "Suite-specific performance")
        lines.append(f"{suite} & {auroc_s} & {eps} & {interp} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write_text(out_path, "\n".join(lines) + "\n")


def build_rollout_index(rollout_dir: Path) -> dict[str, Path]:
    if not rollout_dir.exists():
        return {}
    return {p.stem: p for p in rollout_dir.rglob("rollout_*.safetensors")}


def safe_split_indices(n: int, y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=None)


def compute_motion_control_aurocs(
    labels_df: pd.DataFrame,
    rollout_index: dict[str, Path],
    test_size: float,
    seed: int,
    lr_c: float,
) -> dict[str, float]:
    rows = []
    for _, r in labels_df.iterrows():
        ep = str(r["episode_id"])
        y = int(r["label"])
        path = rollout_index.get(ep)
        if path is None:
            continue
        with safe_open(str(path), framework="np") as f:
            if "actions" not in f.keys() or "eef_positions" not in f.keys():
                continue
            actions = f.get_tensor("actions").astype(np.float64)
            eef = f.get_tensor("eef_positions").astype(np.float64)

        action_mag = np.linalg.norm(actions, axis=1)
        velocity = np.linalg.norm(np.diff(eef, axis=0), axis=1) if eef.shape[0] > 1 else np.array([0.0], dtype=np.float64)
        rows.append(
            {
                "episode_id": ep,
                "label": y,
                "action_magnitude": float(np.mean(action_mag)),
                "eef_velocity": float(np.mean(velocity)),
            }
        )

    if not rows:
        return {
            "action_magnitude_only": np.nan,
            "eef_velocity_only": np.nan,
            "action_magnitude_plus_velocity": np.nan,
        }

    df = pd.DataFrame(rows)
    y = df["label"].to_numpy(dtype=np.int32)
    if len(np.unique(y)) < 2:
        return {
            "action_magnitude_only": 0.5,
            "eef_velocity_only": 0.5,
            "action_magnitude_plus_velocity": 0.5,
        }

    train_idx, test_idx = safe_split_indices(len(df), y, test_size=test_size, seed=seed)

    def _auroc(cols: list[str]) -> float:
        x = df[cols].to_numpy(dtype=np.float64)
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return 0.5
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=lr_c)
        model.fit(x[train_idx], y_train)
        scores = model.predict_proba(x[test_idx])[:, 1]
        return float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5

    return {
        "action_magnitude_only": _auroc(["action_magnitude"]),
        "eef_velocity_only": _auroc(["eef_velocity"]),
        "action_magnitude_plus_velocity": _auroc(["action_magnitude", "eef_velocity"]),
    }


def write_table_motion_control(
    monitor_metrics_path: Path,
    labels_path: Path,
    rollout_index: dict[str, Path],
    out_path: Path,
    test_size: float,
    seed: int,
    lr_c: float,
) -> None:
    monitor = read_json(monitor_metrics_path)
    labels_df = pd.read_csv(labels_path)
    motion = compute_motion_control_aurocs(labels_df, rollout_index, test_size=test_size, seed=seed, lr_c=lr_c)

    sae_full = resolve_metric_row(monitor, ["raw_activation_lr", "sae_lr_16384d", "sae_feature_lr"])["auroc"]
    sae_top20 = resolve_metric_row(monitor, ["top20_feature_lr", "top20_lr"])["auroc"]

    table_rows = [
        ("Action magnitude only", motion["action_magnitude_only"]),
        ("End-effector velocity only", motion["eef_velocity_only"]),
        ("Action magnitude + velocity", motion["action_magnitude_plus_velocity"]),
        ("SAE Feature LR (16384-d)", sae_full),
        ("Top-20 SAE Feature LR", sae_top20),
    ]

    vals = np.array([v for _, v in table_rows], dtype=np.float64)
    best = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan

    lines = [
        "\\begin{tabular}{lc}\\toprule",
        "Control Feature Set & AUROC$\\uparrow$ \\\\ \\midrule",
    ]
    for name, val in table_rows:
        sval = format_float(val)
        is_best = np.isfinite(best) and np.isfinite(val) and abs(float(val) - best) < 1e-12
        lines.append(f"{name} & {bold_if(sval, is_best)} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write_text(out_path, "\n".join(lines) + "\n")


def write_table_dataset_stats(
    labels_full_path: Path,
    labels_path: Path,
    rollout_index: dict[str, Path],
    out_path: Path,
    default_episode_steps: int,
) -> None:
    full_df = pd.read_csv(labels_full_path)
    labels_df = pd.read_csv(labels_path)
    label_map = {str(r["episode_id"]): int(r["label"]) for _, r in labels_df.iterrows()}

    labeled = full_df[full_df["episode_id"].map(lambda x: str(x) in label_map)].copy()
    labeled["label"] = labeled["episode_id"].map(lambda x: label_map[str(x)])

    suite_rows = []
    suite_order = ["goal", "long", "object", "spatial"]
    suites = sorted(labeled["suite"].unique().tolist(), key=lambda s: suite_order.index(s) if s in suite_order else 999)
    for suite in suites:
        sdf = labeled[labeled["suite"] == suite]
        low = int((sdf["label"] == 0).sum())
        high = int((sdf["label"] == 1).sum())
        suite_rows.append((suite, low, high, low + high))

    steps_by_episode: dict[str, int] = {}
    for ep in labeled["episode_id"].astype(str).tolist():
        p = rollout_index.get(ep)
        if p is None:
            continue
        with safe_open(str(p), framework="np") as f:
            if "actions" in f.keys():
                steps_by_episode[ep] = int(f.get_tensor("actions").shape[0])
            elif "eef_positions" in f.keys():
                steps_by_episode[ep] = int(f.get_tensor("eef_positions").shape[0])

    def _episode_steps(ep: str) -> int:
        return int(steps_by_episode.get(ep, default_episode_steps))

    low_steps = int(sum(_episode_steps(ep) for ep in labeled[labeled["label"] == 0]["episode_id"].astype(str).tolist()))
    high_steps = int(sum(_episode_steps(ep) for ep in labeled[labeled["label"] == 1]["episode_id"].astype(str).tolist()))

    low_stats = labeled[labeled["label"] == 0]["progress_norm"]
    high_stats = labeled[labeled["label"] == 1]["progress_norm"]

    low_mean = float(low_stats.mean()) if not low_stats.empty else np.nan
    low_std = float(low_stats.std()) if not low_stats.empty else np.nan
    high_mean = float(high_stats.mean()) if not high_stats.empty else np.nan
    high_std = float(high_stats.std()) if not high_stats.empty else np.nan

    lines = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Suite & Low-progress & High-progress & Total \\\\ \\midrule",
    ]
    for suite, low, high, total in suite_rows:
        lines.append(f"{suite} & {low} & {high} & {total} \\\\")
    lines += [
        "\\midrule",
        f"Low-progress timesteps & \\multicolumn{{3}}{{c}}{{{low_steps}}} \\\\",
        f"High-progress timesteps & \\multicolumn{{3}}{{c}}{{{high_steps}}} \\\\",
        f"Progress norm (low mean$\\pm$std) & \\multicolumn{{3}}{{c}}{{${low_mean:.3f}\\pm{low_std:.3f}$}} \\\\",
        f"Progress norm (high mean$\\pm$std) & \\multicolumn{{3}}{{c}}{{${high_mean:.3f}\\pm{high_std:.3f}$}} \\\\",
        "\\bottomrule\\end{tabular}",
    ]
    write_text(out_path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rollout_index = build_rollout_index(Path(args.rollout_dir))

    write_table_main_results(Path(args.monitor_metrics_json), output_dir / "table_main_results.tex")
    write_table_category_auroc(Path(args.per_suite_csv), output_dir / "table_category_auroc.tex")
    write_table_dataset_stats(
        Path(args.labels_full_csv),
        Path(args.labels_csv),
        rollout_index,
        output_dir / "table_dataset_stats.tex",
        default_episode_steps=int(args.default_episode_steps),
    )
    write_table_motion_control(
        Path(args.monitor_metrics_json),
        Path(args.labels_csv),
        rollout_index,
        output_dir / "table_motion_control.tex",
        test_size=float(args.test_size),
        seed=int(args.seed),
        lr_c=float(args.motion_lr_c),
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tables": [
                    "table_main_results.tex",
                    "table_category_auroc.tex",
                    "table_dataset_stats.tex",
                    "table_motion_control.tex",
                ],
                "rollouts_indexed": len(rollout_index),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
