"""Checkpoint sync helper supporting rsync remotes and AWS S3 targets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync SAE checkpoints to remote storage")
    parser.add_argument("--vast_config", type=str, default="configs/vast_config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def sync_rsync(remote: str, remote_path: str, checkpoint_dir: str) -> None:
    target = f"{remote}:{remote_path.rstrip('/')}/"
    src = f"{Path(checkpoint_dir).as_posix().rstrip('/')}/"
    subprocess.run(["rsync", "-az", "--delete", src, target], check=True)


def sync_s3(s3_uri: str, checkpoint_dir: str) -> None:
    src = f"{Path(checkpoint_dir).as_posix().rstrip('/')}/"
    subprocess.run(["aws", "s3", "sync", src, s3_uri], check=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.vast_config).get("vast", {})

    remote = str(cfg.get("rsync_remote", "")).strip()
    remote_path = str(cfg.get("rsync_path", "")).strip()
    s3_uri = str(cfg.get("s3_uri", "")).strip()

    if remote and remote_path:
        sync_rsync(remote, remote_path, args.checkpoint_dir)
    if s3_uri:
        sync_s3(s3_uri, args.checkpoint_dir)


if __name__ == "__main__":
    main()
