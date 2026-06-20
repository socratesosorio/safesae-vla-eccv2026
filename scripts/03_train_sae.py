"""Train all SAEs for OpenVLA and optional pi0 generalization runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sae.train import train_sae
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SafeSAE-VLA SAE checkpoints")
    parser.add_argument("--openvla_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--openvla_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sae")

    parser.add_argument("--pi0_config", type=str, default="configs/sae_pi0.yaml")
    parser.add_argument("--pi0_data_dir", type=str, default="")
    parser.add_argument("--skip_pi0", action="store_true")
    return parser.parse_args()


def _train_openvla(args: argparse.Namespace) -> list[dict]:
    cfg = load_yaml(args.openvla_config)
    primary = cfg.get("primary", cfg)
    ablation_32k = cfg.get("ablation_32k", {})
    d_sae_main = int(primary.get("d_sae", 16384))
    k_main = int(primary.get("k", 32))

    out = []
    for layer in [16, 20, 24]:
        result = train_sae(
            config_path=args.openvla_config,
            data_dir=args.openvla_data_dir,
            output_dir=args.output_dir,
            layer=layer,
            d_sae=d_sae_main,
            k=k_main,
        )
        out.append(result)

    d_sae_ab = int(ablation_32k.get("d_sae", 32768))
    k_ab = int(ablation_32k.get("k", 48))
    out.append(
        train_sae(
            config_path=args.openvla_config,
            data_dir=args.openvla_data_dir,
            output_dir=args.output_dir,
            layer=20,
            d_sae=d_sae_ab,
            k=k_ab,
        )
    )
    return out


def _train_pi0(args: argparse.Namespace) -> list[dict]:
    pi0_dir = str(args.pi0_data_dir).strip()
    if args.skip_pi0 or not pi0_dir:
        return []
    if not Path(pi0_dir).exists():
        return []

    cfg = load_yaml(args.pi0_config)
    primary = cfg.get("primary", cfg)
    d_sae = int(primary.get("d_sae", 8192))
    k = int(primary.get("k", 32))

    out = []
    for layer in [9, 11, 14]:
        result = train_sae(
            config_path=args.pi0_config,
            data_dir=pi0_dir,
            output_dir=args.output_dir,
            layer=layer,
            d_sae=d_sae,
            k=k,
        )
        out.append(result)
    return out


def main() -> None:
    args = parse_args()
    results = {
        "openvla": _train_openvla(args),
        "pi0": _train_pi0(args),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
