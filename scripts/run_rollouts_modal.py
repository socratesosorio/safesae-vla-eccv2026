"""Modal deployment script for parallel rollout collection and local fallback."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml

DEFAULT_MODAL_CFG = {
    "app_name": "safesae-vla-rollouts",
    "volume_name": "safesae-vla-data",
    "gpu_type": "A100",
    "gpu_size": "80GB",
    "num_workers": 8,
    "timeout_sec": 86400,
    "retries": 2,
    "retry_backoff_sec": 15,
    "mount_path": "/data",
    "startup_timeout_sec": 900,
}


def load_modal_cfg(path: str = "configs/modal_config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return DEFAULT_MODAL_CFG.copy()
    data = load_yaml(cfg_path)
    out = DEFAULT_MODAL_CFG.copy()
    out.update(data.get("modal", {}))
    return out


app = modal.App(DEFAULT_MODAL_CFG["app_name"])

# Pre-warm model snapshots into the image to reduce cold starts.
_PRELOAD_SNIPPET = r"""
python - <<'PY'
from huggingface_hub import snapshot_download
models = [
  'openvla/openvla-7b',
  'openvla/openvla-7b-finetuned-libero-spatial',
  'openvla/openvla-7b-finetuned-libero-object',
  'openvla/openvla-7b-finetuned-libero-goal',
  'openvla/openvla-7b-finetuned-libero-10',
]
for m in models:
    snapshot_download(repo_id=m, repo_type='model', local_files_only=False)
print('preloaded models')
PY
"""

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.2.0",
        "transformers>=4.40.0",
        "accelerate",
        "safetensors",
        "robosuite==1.5.0",
        "mujoco>=3.0",
        "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "numpy",
        "Pillow",
        "pyyaml",
        "tqdm",
        "huggingface_hub>=0.22.0",
    )
    .run_commands(_PRELOAD_SNIPPET)
)

def _gpu_spec(cfg: dict):
    gpu_type = str(cfg.get("gpu_type", "A100")).upper()
    gpu_size = str(cfg.get("gpu_size", "80GB"))
    if gpu_type == "A100":
        return modal.gpu.A100(size=gpu_size)
    if gpu_type == "H100":
        return modal.gpu.H100(size=gpu_size)
    return modal.gpu.A100(size="80GB")


def _collect_rollouts_chunk_impl(chunk_id: int, total_chunks: int, config: dict, modal_cfg: dict):
    from src.data.rollout_collector import RolloutCollector

    collector = RolloutCollector(config)
    total = int(config["collection"]["total_rollouts"])
    chunk_size = total // total_chunks
    start = chunk_id * chunk_size
    end = total if chunk_id == total_chunks - 1 else start + chunk_size

    retries = int(modal_cfg.get("retries", 2))
    backoff = int(modal_cfg.get("retry_backoff_sec", 15))
    mount_path = str(modal_cfg.get("mount_path", "/data"))
    volume_name = str(modal_cfg.get("volume_name", DEFAULT_MODAL_CFG["volume_name"]))
    volume_handle = modal.Volume.from_name(volume_name, create_if_missing=True)

    for attempt in range(retries + 1):
        try:
            collector.collect_range(
                start_idx=start,
                end_idx=end,
                output_dir=f"{mount_path}/rollouts/chunk_{chunk_id}",
            )
            volume_handle.commit()
            return {"chunk_id": chunk_id, "status": "ok", "attempt": attempt + 1}
        except Exception as exc:
            if attempt >= retries:
                raise
            time.sleep(backoff * (attempt + 1))


def _build_collect_rollouts_fn(modal_cfg: dict):
    mount_path = str(modal_cfg.get("mount_path", DEFAULT_MODAL_CFG["mount_path"]))
    volume_name = str(modal_cfg.get("volume_name", DEFAULT_MODAL_CFG["volume_name"]))
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return app.function(
        image=image,
        gpu=_gpu_spec(modal_cfg),
        timeout=int(modal_cfg.get("timeout_sec", DEFAULT_MODAL_CFG["timeout_sec"])),
        volumes={mount_path: volume},
    )(_collect_rollouts_chunk_impl)


def _resolve_worker_count(num_workers: int | None, modal_cfg: dict) -> int:
    return int(modal_cfg.get("num_workers", DEFAULT_MODAL_CFG["num_workers"]) if num_workers is None else num_workers)


def load_rollout_config(path: str) -> dict:
    return load_yaml(path)


@app.local_entrypoint()
def modal_main(
    config_path: str = "configs/rollout.yaml",
    modal_config_path: str = "configs/modal_config.yaml",
    num_workers: int | None = None,
):
    config = load_rollout_config(config_path)
    modal_cfg = load_modal_cfg(modal_config_path)
    workers = _resolve_worker_count(num_workers, modal_cfg)
    collect_rollouts_chunk = _build_collect_rollouts_fn(modal_cfg)

    futures = [collect_rollouts_chunk.spawn(i, workers, config, modal_cfg) for i in range(workers)]
    for fut in futures:
        print(fut.get())
    print("All rollouts collected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout collection on Modal or local")
    parser.add_argument("--mode", type=str, default="modal", choices=["modal", "local"])
    parser.add_argument("--config", type=str, default="configs/rollout.yaml")
    parser.add_argument("--modal_config", type=str, default="configs/modal_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/rollouts")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "local":
        from src.data.rollout_collector import RolloutCollector

        cfg = load_rollout_config(args.config)
        collector = RolloutCollector(cfg)
        collector.collect_all(args.output_dir)
        return

    print(
        "Use Modal CLI for remote execution: modal run scripts/run_rollouts_modal.py "
        f"--config-path {args.config} --modal-config-path {args.modal_config} "
        f"--num-workers {args.num_workers}"
    )


if __name__ == "__main__":
    main()
