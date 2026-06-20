"""Closed-loop evaluation of (optionally LoRA-fine-tuned) OpenVLA on LIBERO.

Modes:
  clean eval:       --suite object --tasks 0-9 --episodes 30-39 [--adapter DIR]
  perturbed eval:   add --perturb-spec name:type:param (DreamAudit observation grammar)
  cert replay:      --certs-json certs.json [--adapter DIR]
                    (replays each certificate's exact task/init-state/perturbation)

Writes one JSONL row per episode and a summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.rl4vla_loops.frame_collector import (  # noqa: E402
    FrameCollector,
    parse_observation_perturbation_spec,
)
from scripts.rl4vla_loops.collect_frames import parse_range  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


def episode_row(rollout: dict, *, spec: dict, condition: str, adapter: str | None) -> dict:
    meta = rollout["metadata"]
    return {
        "suite": meta["suite"],
        "task_idx": int(meta["task_idx"]),
        "episode_idx": int(meta.get("episode_idx", -1)),
        "init_state_index": meta.get("init_state_index"),
        "instruction": meta["instruction"],
        "condition": condition,
        "perturbation": spec,
        "adapter": adapter,
        "success": bool(meta["episode_success"]),
        "success_by_info": bool(meta["episode_success_by_info"]),
        "success_by_env_check": bool(meta["episode_success_by_env_check"]),
        "success_by_done": bool(meta["episode_success_by_done"]),
        "num_steps": int(meta["num_steps"]),
        "time": time.time(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", default="object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", default="30-39")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--perturb-spec", default=None)
    ap.add_argument("--certs-json", default=None)
    ap.add_argument("--cert-start", type=int, default=0)
    ap.add_argument("--cert-end", type=int, default=10**9)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    config = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{args.tag}.jsonl"
    done_keys = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add((r["suite"], r["task_idx"], r["episode_idx"], r["perturbation"].get("name", "native")))
            except Exception:
                continue
    rows_f = rows_path.open("a")

    jobs: list[tuple[str, int, int, dict]] = []
    if args.certs_json:
        certs = json.loads(Path(args.certs_json).read_text())["certificates"]
        for c in certs[args.cert_start : args.cert_end]:
            spec = {"name": c["name"], "type": c["type"], "params": c["params"]}
            jobs.append((str(c["suite"]), int(c["task_idx"]), int(c["episode_idx"]), spec))
    else:
        spec = (
            parse_observation_perturbation_spec(args.perturb_spec)
            if args.perturb_spec
            else {"name": "native", "type": "identity", "params": {}}
        )
        for task_idx in parse_range(args.tasks):
            for ep_idx in parse_range(args.episodes):
                jobs.append((args.suite, task_idx, ep_idx, spec))

    collector = None
    current_spec_key = None
    n, n_succ = 0, 0
    for suite, task_idx, ep_idx, spec in jobs:
        key = (suite, task_idx, ep_idx, spec.get("name", "native"))
        if key in done_keys:
            print(f"[skip] {key}")
            continue
        spec_key = json.dumps(spec, sort_keys=True)
        if collector is None or spec_key != current_spec_key:
            collector = FrameCollector(
                config, save_frames=False, obs_perturb_spec=spec, adapter_dir=args.adapter
            ) if collector is None else collector
            collector._obs_spec = spec  # reuse loaded model across specs
            current_spec_key = spec_key
        t0 = time.time()
        rollout = collector.collect_episode(suite, task_idx, ep_idx)
        row = episode_row(rollout, spec=spec, condition=args.condition, adapter=args.adapter)
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()
        n += 1
        n_succ += int(row["success"])
        print(
            f"[{n}/{len(jobs)}] {suite} t{task_idx} e{ep_idx} {spec.get('name','native')} "
            f"success={row['success']} dt={time.time()-t0:.1f}s",
            flush=True,
        )

    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "episodes_evaluated": n,
        "successes": n_succ,
        "rows_file": str(rows_path),
    }
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
