"""Cloud preflight validation for Modal/Vast experiment execution."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cloud/runtime setup before long jobs")
    parser.add_argument("--provider", type=str, default="both", choices=["modal", "vast", "both"])
    parser.add_argument("--rollout_config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--sae_config", type=str, default="configs/sae.yaml")
    parser.add_argument("--modal_config", type=str, default="configs/modal_config.yaml")
    parser.add_argument("--vast_config", type=str, default="configs/vast_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    return parser.parse_args()


def exists_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def validate_modal(modal_cfg: dict, out: dict) -> None:
    errs = []
    warns = []

    if not exists_cmd("modal"):
        warns.append("`modal` CLI not found in PATH")

    required = ["app_name", "volume_name", "gpu_type", "num_workers", "timeout_sec"]
    for key in required:
        if key not in modal_cfg:
            errs.append(f"modal config missing key: {key}")

    workers = int(modal_cfg.get("num_workers", 0))
    if workers <= 0:
        errs.append("modal.num_workers must be > 0")

    timeout = int(modal_cfg.get("timeout_sec", 0))
    if timeout < 1800:
        warns.append("modal.timeout_sec is low for rollout collection")

    out["modal"] = {
        "errors": errs,
        "warnings": warns,
        "ok": len(errs) == 0,
    }


def validate_vast(vast_cfg: dict, sae_cfg: dict, out: dict) -> None:
    errs = []
    warns = []

    data_dir = str(vast_cfg.get("data_dir", "")).strip()
    ckpt_dir = str(vast_cfg.get("checkpoint_dir", "")).strip()
    if not data_dir:
        warns.append("vast.data_dir empty; script fallback path will be used")
    if not ckpt_dir:
        warns.append("vast.checkpoint_dir empty; script fallback path will be used")

    rsync_remote = str(vast_cfg.get("rsync_remote", "")).strip()
    rsync_path = str(vast_cfg.get("rsync_path", "")).strip()
    s3_uri = str(vast_cfg.get("s3_uri", "")).strip()

    if (rsync_remote and not rsync_path) or (rsync_path and not rsync_remote):
        errs.append("vast rsync settings must specify both rsync_remote and rsync_path")
    if rsync_remote and not exists_cmd("rsync"):
        errs.append("rsync sync configured but `rsync` command is unavailable")

    if s3_uri and not exists_cmd("aws"):
        errs.append("S3 sync configured but `aws` CLI is unavailable")

    if int(vast_cfg.get("checkpoint_sync_interval_min", 0)) <= 0:
        errs.append("vast.checkpoint_sync_interval_min must be > 0")

    tr_cfg = sae_cfg.get("training", {})
    ckpt_interval = int(tr_cfg.get("checkpoint_interval", tr_cfg.get("checkpoint_every_steps", 0)))
    if ckpt_interval <= 0:
        errs.append("sae training checkpoint_interval must be > 0")
    elif ckpt_interval > 50000:
        warns.append("checkpoint_interval is large; interruption risk increases on interruptible instances")

    out["vast"] = {
        "errors": errs,
        "warnings": warns,
        "ok": len(errs) == 0,
    }


def main() -> None:
    args = parse_args()

    rollout_cfg = load_yaml(args.rollout_config)
    sae_cfg = load_yaml(args.sae_config)

    summary: dict[str, dict] = {
        "rollout_config_loaded": {"ok": bool(rollout_cfg)},
        "sae_config_loaded": {"ok": bool(sae_cfg)},
    }

    if args.provider in {"modal", "both"}:
        modal_cfg = load_yaml(args.modal_config).get("modal", {})
        validate_modal(modal_cfg, summary)

    if args.provider in {"vast", "both"}:
        vast_cfg = load_yaml(args.vast_config).get("vast", {})
        validate_vast(vast_cfg, sae_cfg, summary)

    all_errors = []
    all_warnings = []
    for v in summary.values():
        if isinstance(v, dict):
            all_errors.extend(v.get("errors", []))
            all_warnings.extend(v.get("warnings", []))

    summary["final"] = {
        "num_errors": len(all_errors),
        "num_warnings": len(all_warnings),
        "status": "ok" if (len(all_errors) == 0 and (len(all_warnings) == 0 or not args.strict)) else "failed",
    }

    print(json.dumps(summary, indent=2))

    if all_errors:
        raise SystemExit(1)
    if args.strict and all_warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
