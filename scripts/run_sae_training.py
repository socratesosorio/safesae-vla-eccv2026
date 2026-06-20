"""Launcher for SAE training with local or Vast.ai-oriented workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SAE training")
    parser.add_argument("--backend", type=str, default="manual", choices=["saelens", "manual"])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/sae.yaml")
    parser.add_argument("--vast_config", type=str, default="configs/vast_config.yaml")
    parser.add_argument("--data_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "vast"])
    parser.add_argument("--sync_once", action="store_true")
    return parser.parse_args()


def build_train_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "-m",
        "src.sae.train_sae",
        "--backend",
        args.backend,
        "--layer",
        str(args.layer),
        "--config",
        args.config,
        "--data_dir",
        args.data_dir,
        "--checkpoint_dir",
        args.checkpoint_dir,
    ]
    if args.resume:
        cmd.append("--resume")
    if args.resume_path:
        cmd.extend(["--resume_path", args.resume_path])
    return cmd


def maybe_sync_checkpoints(vast_cfg: dict, checkpoint_dir: str) -> None:
    remote = str(vast_cfg.get("rsync_remote", "")).strip()
    remote_path = str(vast_cfg.get("rsync_path", "")).strip()
    if not remote or not remote_path:
        return

    target = f"{remote}:{remote_path.rstrip('/')}/"
    src = f"{Path(checkpoint_dir).as_posix().rstrip('/')}/"
    subprocess.run(["rsync", "-az", "--delete", src, target], check=True)


def main() -> None:
    args = parse_args()
    train_cmd = build_train_cmd(args)

    if args.provider == "local":
        subprocess.run(train_cmd, check=True)
        return

    vast_cfg = load_yaml(args.vast_config).get("vast", {})
    install_cmd = str(vast_cfg.get("install_cmd", "")).strip()
    sync_interval = int(vast_cfg.get("checkpoint_sync_interval_min", 30))

    if install_cmd:
        subprocess.run(shlex.split(install_cmd), check=True)

    if args.sync_once:
        maybe_sync_checkpoints(vast_cfg, args.checkpoint_dir)
        return

    proc = subprocess.Popen(train_cmd)
    try:
        last_sync = 0.0
        while proc.poll() is None:
            now = time.time()
            if now - last_sync > sync_interval * 60:
                try:
                    maybe_sync_checkpoints(vast_cfg, args.checkpoint_dir)
                except Exception:
                    pass
                last_sync = now
            time.sleep(10)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, train_cmd)
        maybe_sync_checkpoints(vast_cfg, args.checkpoint_dir)
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
