"""
Train 32K-ablation SAE on a given layer.
Designed for cluster use: requires activation data directory.
Usage:
    python scripts/train_32k_ablation.py --data_dir /path/to/rollouts --layer 20 --output_dir outputs/sae_32k

On macOS with MPS:
    PYTORCH_MPS_DISABLE_OPS=expand python scripts/train_32k_ablation.py --data_dir /path/to/rollouts
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.sae.train import train_sae


def main():
    parser = argparse.ArgumentParser(description="Train 32K dictionary-size ablation SAE")
    parser.add_argument("--data_dir", required=True, help="Directory with rollout_*.safetensors")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/sae_32k")
    args = parser.parse_args()

    config_path = str(REPO / "configs" / "sae.yaml")
    metrics = train_sae(
        config_path=config_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layer=args.layer,
        d_sae=32768,
        k=48,
    )
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
