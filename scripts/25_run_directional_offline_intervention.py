"""Offline sign-aware intervention test for progress SAE features.

This complements closed-loop simulator sweeps with a larger cached-activation
test. It asks whether feature signs are directionally meaningful: suppressing
features higher in low-progress episodes and amplifying features higher in
high-progress episodes should move a held-out progress probe differently from
the opposite sign intervention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
ASAP_PATH = ROOT / "scripts" / "23_run_asap_progress_experiments.py"


def _load_asap_module():
    spec = importlib.util.spec_from_file_location("asap_progress_experiments", ASAP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {ASAP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asap = _load_asap_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--labels_full_csv", type=str, required=True)
    p.add_argument("--top_features_csv", type=str, required=True)
    p.add_argument("--sae_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="logs/directional_offline_intervention")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d_in", type=int, default=4096)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--features_per_sign", type=int, default=5)
    p.add_argument("--amplify", type=float, default=1.5)
    p.add_argument("--max_timesteps_per_episode", type=int, default=8)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--random_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_scale(x: np.ndarray, scale_map: dict[int, float]) -> np.ndarray:
    y = x.copy()
    for idx, scale in scale_map.items():
        y[:, int(idx)] *= float(scale)
    return y


def summarize(
    *,
    name: str,
    x: np.ndarray,
    raw: np.ndarray,
    scaled: np.ndarray,
    active_cols: np.ndarray,
    scaler,
    probe,
    raw_scaler,
    action_readout,
    sae,
    norm_factor: float,
) -> dict[str, float | str | int]:
    raw_delta = asap.decoded_delta(sae, x, scaled, norm_factor=norm_factor)
    before_logit, before_action = asap.model_outputs(x, raw, active_cols, scaler, probe, raw_scaler, action_readout)
    after_logit, after_action = asap.model_outputs(
        scaled,
        raw + raw_delta,
        active_cols,
        scaler,
        probe,
        raw_scaler,
        action_readout,
    )
    logit_delta = after_logit - before_logit
    action_shift = np.linalg.norm(after_action - before_action, axis=1)
    hidden_shift = np.linalg.norm(raw_delta, axis=1)
    return {
        "condition": name,
        "n_samples": int(x.shape[0]),
        "mean_progress_logit_delta": float(np.mean(logit_delta)),
        "median_progress_logit_delta": float(np.median(logit_delta)),
        "frac_progress_logit_increased": float(np.mean(logit_delta > 0.0)),
        "mean_abs_progress_logit_delta": float(np.mean(np.abs(logit_delta))),
        "mean_raw_action_readout_shift_l2": float(np.mean(action_shift)),
        "median_raw_action_readout_shift_l2": float(np.median(action_shift)),
        "mean_raw_activation_delta_l2": float(np.mean(hidden_shift)),
        "median_raw_activation_delta_l2": float(np.median(hidden_shift)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_full = pd.read_csv(args.labels_full_csv)
    top_df = pd.read_csv(args.top_features_csv)
    low_features = (
        top_df[top_df["direction"] == "higher_in_low_progress"]["feature_idx"]
        .astype(int)
        .head(args.features_per_sign)
        .tolist()
    )
    high_features = (
        top_df[top_df["direction"] == "higher_in_high_progress"]["feature_idx"]
        .astype(int)
        .head(args.features_per_sign)
        .tolist()
    )
    if len(low_features) < args.features_per_sign or len(high_features) < args.features_per_sign:
        raise ValueError("Not enough direction-labeled top features for requested features_per_sign")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae, norm_factor = asap.load_sae_checkpoint(
        args.sae_checkpoint,
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        device=device,
    )
    bundle = asap.load_samples(
        data_dir=Path(args.data_dir),
        labels_full=labels_full,
        layer=args.layer,
        sae=sae,
        norm_factor=norm_factor,
        device=device,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        max_episodes=args.max_episodes,
    )

    active_cols = np.flatnonzero((bundle.features > 0).any(axis=0))
    _, test_idx, scaler, probe, raw_scaler, action_readout, diagnostics = asap.fit_models(
        bundle,
        active_cols,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    active_pool = np.asarray([i for i in active_cols if i not in set(low_features + high_features)], dtype=np.int64)

    help_map = {idx: 0.0 for idx in low_features} | {idx: float(args.amplify) for idx in high_features}
    harm_map = {idx: float(args.amplify) for idx in low_features} | {idx: 0.0 for idx in high_features}
    low_zero_map = {idx: 0.0 for idx in low_features}
    high_amp_map = {idx: float(args.amplify) for idx in high_features}

    rows: list[dict[str, float | str | int]] = []
    subset_specs = {
        "all_test": test_idx,
        "low_progress_test": test_idx[bundle.labels[test_idx] == 0],
        "high_progress_test": test_idx[bundle.labels[test_idx] == 1],
    }
    for subset_name, idx in subset_specs.items():
        if len(idx) == 0:
            continue
        x = bundle.features[idx]
        raw = bundle.raw_activations[idx]
        for name, scale_map in [
            ("directional_help", help_map),
            ("directional_harm", harm_map),
            ("low_only_zero", low_zero_map),
            ("high_only_amplify", high_amp_map),
        ]:
            row = summarize(
                name=f"{subset_name}_{name}",
                x=x,
                raw=raw,
                scaled=apply_scale(x, scale_map),
                active_cols=active_cols,
                scaler=scaler,
                probe=probe,
                raw_scaler=raw_scaler,
                action_readout=action_readout,
                sae=sae,
                norm_factor=norm_factor,
            )
            row["subset"] = subset_name
            rows.append(row)

        random_rows = []
        for trial in range(max(int(args.random_trials), 1)):
            chosen = rng.choice(active_pool, size=len(low_features) + len(high_features), replace=False)
            random_map = {int(idx): 0.0 for idx in chosen[: len(low_features)]}
            random_map.update({int(idx): float(args.amplify) for idx in chosen[len(low_features) :]})
            random_rows.append(
                summarize(
                    name=f"{subset_name}_random_mixed_trial{trial:03d}",
                    x=x,
                    raw=raw,
                    scaled=apply_scale(x, random_map),
                    active_cols=active_cols,
                    scaler=scaler,
                    probe=probe,
                    raw_scaler=raw_scaler,
                    action_readout=action_readout,
                    sae=sae,
                    norm_factor=norm_factor,
                )
            )
        rand = pd.DataFrame(random_rows)
        mean_row = {
            "condition": f"{subset_name}_random_mixed_mean",
            "subset": subset_name,
            "n_samples": int(len(idx)),
        }
        for col in rand.columns:
            if col not in {"condition", "n_samples"}:
                mean_row[col] = float(rand[col].mean())
        rows.append(mean_row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "directional_offline_intervention.csv", index=False)

    summary = {
        **diagnostics,
        "num_samples": int(bundle.features.shape[0]),
        "num_test_samples": int(len(test_idx)),
        "low_progress_features": low_features,
        "high_progress_features": high_features,
        "amplify": float(args.amplify),
        "features_per_sign": int(args.features_per_sign),
    }
    all_rows = df[df["subset"] == "all_test"].set_index("condition")
    help_key = "all_test_directional_help"
    harm_key = "all_test_directional_harm"
    if help_key in all_rows.index and harm_key in all_rows.index:
        summary["all_test_help_minus_harm_logit_delta"] = float(
            all_rows.loc[help_key, "mean_progress_logit_delta"]
            - all_rows.loc[harm_key, "mean_progress_logit_delta"]
        )
        summary["all_test_help_minus_random_logit_delta"] = float(
            all_rows.loc[help_key, "mean_progress_logit_delta"]
            - all_rows.loc["all_test_random_mixed_mean", "mean_progress_logit_delta"]
        )
    with (out_dir / "directional_offline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
