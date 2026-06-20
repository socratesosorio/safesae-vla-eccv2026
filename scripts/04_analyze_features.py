"""Run multi-layer differential analysis, ablations, and cross-model comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.cross_model_comparison import CrossModelComparison
from src.analysis.differential_activation import (
    DifferentialActivationAnalyzer,
    load_sae_checkpoint,
)
from src.data.activation_dataset import AnalysisDataset
from src.utils.config import load_yaml
from src.utils.runtime import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze safety features across models/layers")
    parser.add_argument("--openvla_data_dir", type=str, required=True)
    parser.add_argument("--openvla_sae_dir", type=str, required=True)
    parser.add_argument("--openvla_sae_config", type=str, default="configs/sae.yaml")

    parser.add_argument("--pi0_data_dir", type=str, default="")
    parser.add_argument("--pi0_sae_dir", type=str, default="")
    parser.add_argument("--pi0_sae_config", type=str, default="configs/sae_pi0.yaml")
    parser.add_argument("--skip_pi0", action="store_true")

    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    parser.add_argument(
        "--progress_labels_csv",
        type=str,
        default="",
        help="Optional CSV with episode_id,label (1=high_progress, 0=low_progress)",
    )
    parser.add_argument("--skip_activation_patching", action="store_true")
    return parser.parse_args()


def _sae_dims(cfg: dict) -> tuple[int, int, int]:
    primary = cfg.get("primary", cfg.get("sae", cfg))
    return int(primary.get("d_in", 4096)), int(primary.get("d_sae", 16384)), int(primary.get("k", 32))


def _run_model_analysis(
    data_dir: str,
    sae_dir: str,
    sae_config_path: str,
    eval_cfg: dict,
    layers: list[int],
    ckpt_template: str,
    progress_labels_csv: str = "",
) -> dict[int, dict[str, pd.DataFrame]]:
    dataset = AnalysisDataset(data_dir=data_dir, test_split=float(eval_cfg.get("analysis", {}).get("test_split", 0.2)))
    sae_cfg = load_yaml(sae_config_path)
    d_in, d_sae_default, k_default = _sae_dims(sae_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[int, dict[str, pd.DataFrame]] = {}
    episode_labels: dict[str, int] | None = None
    if str(progress_labels_csv).strip():
        labels_df = pd.read_csv(progress_labels_csv)
        if "episode_id" not in labels_df.columns or "label" not in labels_df.columns:
            raise ValueError("progress_labels_csv must contain columns: episode_id,label")
        episode_labels = {
            str(row["episode_id"]): int(row["label"])
            for _, row in labels_df.iterrows()
        }
    for layer in layers:
        ckpt_path = Path(sae_dir) / ckpt_template.format(layer=layer, d_sae=d_sae_default)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing SAE checkpoint: {ckpt_path}")
        sae, norm_factor = load_sae_checkpoint(str(ckpt_path), d_in=d_in, d_sae=d_sae_default, k=k_default, device=device)
        analyzer = DifferentialActivationAnalyzer(sae=sae, config=eval_cfg, norm_factor=norm_factor)
        results[layer] = analyzer.run_layer_analysis(
            dataset=dataset,
            layer=layer,
            episode_labels=episode_labels,
        )
    return results


def _save_layer_results(base_dir: Path, prefix: str, layer_results: dict[int, dict[str, pd.DataFrame]]) -> None:
    ensure_dir(base_dir)
    for layer, named in layer_results.items():
        for name, df in named.items():
            df.to_csv(base_dir / f"{prefix}_layer{layer}_{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.eval_config)
    ov_cfg = load_yaml(args.openvla_sae_config)
    ov_primary = ov_cfg.get("primary", ov_cfg)
    ov_d_sae = int(ov_primary.get("d_sae", 16384))
    out_root = ensure_dir(args.output_dir)
    diff_dir = ensure_dir(Path(out_root) / "differential")

    openvla_results = _run_model_analysis(
        data_dir=args.openvla_data_dir,
        sae_dir=args.openvla_sae_dir,
        sae_config_path=args.openvla_sae_config,
        eval_cfg=eval_cfg,
        layers=[16, 20, 24],
        ckpt_template="sae_layer{layer}_d{d_sae}.pt",
        progress_labels_csv=args.progress_labels_csv,
    )
    _save_layer_results(Path(diff_dir), "openvla", openvla_results)

    if not args.skip_activation_patching:
        ranked_features = Path(diff_dir) / "openvla_layer20_overall.csv"
        subprocess.run(
            [
                "python",
                "-m",
                "src.analysis.activation_patching",
                "--data_dir",
                args.openvla_data_dir,
                "--sae_checkpoint",
                str(Path(args.openvla_sae_dir) / f"sae_layer20_d{ov_d_sae}.pt"),
                "--ranked_features",
                str(ranked_features),
                "--sae_config",
                args.openvla_sae_config,
                "--eval_config",
                args.eval_config,
                "--output_dir",
                str(Path(out_root) / "activation_patching"),
                "--layer",
                "20",
            ],
            check=True,
        )

    # Dictionary-size ablation: layer 20, d=32768, k=48.
    ablation_ckpt = Path(args.openvla_sae_dir) / "sae_layer20_d32768.pt"
    if not ablation_ckpt.exists():
        raise FileNotFoundError(f"Missing OpenVLA 32K ablation checkpoint: {ablation_ckpt}")
    ov_d_in = int(ov_cfg.get("primary", ov_cfg).get("d_in", 4096))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_32k, norm_32k = load_sae_checkpoint(str(ablation_ckpt), d_in=ov_d_in, d_sae=32768, k=48, device=device)
    ov_dataset = AnalysisDataset(
        data_dir=args.openvla_data_dir,
        test_split=float(eval_cfg.get("analysis", {}).get("test_split", 0.2)),
    )
    analyzer_32k = DifferentialActivationAnalyzer(sae=sae_32k, config=eval_cfg, norm_factor=norm_32k)
    ablation_32k = analyzer_32k.run_layer_analysis(
        ov_dataset,
        layer=20,
        episode_labels=None if not str(args.progress_labels_csv).strip() else {
            str(r["episode_id"]): int(r["label"])
            for _, r in pd.read_csv(args.progress_labels_csv).iterrows()
        },
    )
    for name, df in ablation_32k.items():
        df.to_csv(Path(diff_dir) / f"openvla_d32768_layer20_{name}.csv", index=False)

    # Dict-size summary artifact.
    o16 = openvla_results[20]["overall"]
    o32 = ablation_32k["overall"]
    ablation_summary = {
        "16k_significant_features": int(o16["significant"].astype(bool).sum()) if not o16.empty and "significant" in o16 else 0,
        "32k_significant_features": int(o32["significant"].astype(bool).sum()) if not o32.empty and "significant" in o32 else 0,
        "16k_top10_mean_effect": float(o16["abs_effect_size"].head(10).mean()) if not o16.empty and "abs_effect_size" in o16 else 0.0,
        "32k_top10_mean_effect": float(o32["abs_effect_size"].head(10).mean()) if not o32.empty and "abs_effect_size" in o32 else 0.0,
    }
    with (Path(out_root) / "dict_size_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    layer_comparison = DifferentialActivationAnalyzer.compare_layers(openvla_results)
    with (Path(out_root) / "layer_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(layer_comparison, f, indent=2)

    # pi0 optional path.
    pi0_results: dict[int, dict[str, pd.DataFrame]] = {}
    if not args.skip_pi0 and str(args.pi0_data_dir).strip() and Path(args.pi0_data_dir).exists():
        sae_dir = args.pi0_sae_dir if args.pi0_sae_dir else args.openvla_sae_dir
        pi0_results = _run_model_analysis(
            data_dir=args.pi0_data_dir,
            sae_dir=sae_dir,
            sae_config_path=args.pi0_sae_config,
            eval_cfg=eval_cfg,
            layers=[9, 11, 14],
            ckpt_template="sae_layer{layer}_d{d_sae}.pt",
            progress_labels_csv="",
        )
        _save_layer_results(Path(diff_dir), "pi0", pi0_results)

        comparator = CrossModelComparison(openvla_results=openvla_results, pi0_results=pi0_results)
        structural = comparator.structural_comparison()
        with (Path(out_root) / "cross_model_comparison.json").open("w", encoding="utf-8") as f:
            serializable = dict(structural)
            serializable["category_comparison"] = structural["category_comparison"].to_dict(orient="records")
            json.dump(serializable, f, indent=2)
        comparator.generate_comparison_figures(output_dir=str(Path(out_root) / "figures"))

    print(
        json.dumps(
            {
                "openvla_layers": sorted(openvla_results.keys()),
                "pi0_layers": sorted(pi0_results.keys()),
                "output_dir": str(out_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
