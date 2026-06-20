"""Collect OpenVLA LIBERO rollouts with frames + action tokens for the RL4VLA loops.

Example (one chunk):
  python scripts/rl4vla_loops/collect_frames.py \
    --config scripts/rl4vla_loops/rl4vla_collect.yaml \
    --suite object --tasks 0-9 --episodes 0-29 --task-stride 1 \
    --output_dir /work/joy/rl4vla_loops/data/object_train
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.rl4vla_loops.frame_collector import FrameCollector  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.runtime import save_json  # noqa: E402


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", default="object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", default="0-29", help="episode/init-state indices, e.g. 0-29 or 30-39")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no-frames", action="store_true", help="skip frame/token saving (eval-style collection)")
    args = ap.parse_args()

    config = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collector = FrameCollector(config, save_frames=not args.no_frames)

    tasks = parse_range(args.tasks)
    episodes = parse_range(args.episodes)
    started = time.time()
    n_done = 0
    n_succ = 0
    for task_idx in tasks:
        for ep_idx in episodes:
            rollout_id = f"{args.suite}_t{task_idx:02d}_e{ep_idx:03d}"
            if (out_dir / f"{rollout_id}.json").exists():
                print(f"[skip] {rollout_id} exists")
                continue
            t0 = time.time()
            rollout = collector.collect_episode(args.suite, task_idx, ep_idx)
            collector.save_episode(rollout, out_dir, rollout_id)
            n_done += 1
            n_succ += int(bool(rollout["episode_success"]))
            print(
                f"[{n_done}] {rollout_id} success={bool(rollout['episode_success'])} "
                f"steps={rollout['actions'].shape[0]} dt={time.time()-t0:.1f}s",
                flush=True,
            )

    save_json(
        out_dir / f"collect_summary_{args.suite}_{args.tasks}_{args.episodes}.json".replace(",", "_"),
        {
            "suite": args.suite,
            "tasks": tasks,
            "episodes": episodes,
            "collected": n_done,
            "successes": n_succ,
            "elapsed_s": time.time() - started,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
