"""Assemble replayable failure certificates from perturbed-eval JSONL rows.

A certificate = (suite, task_idx, episode_idx/init-state, perturbation spec) where the
clean episode succeeded (native anchor) and the perturbed replay failed. Mirrors the
DreamAudit counterfactual-certificate definition on the observation channel.

Usage:
  python build_certs.py --clean clean.jsonl --perturbed mine_*.jsonl \
      --out certs.json [--split-episodes-leq 29]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_rows(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for p in glob.glob(pat):
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="*", default=[], help="clean (native) eval jsonl(s)")
    ap.add_argument("--clean-collect-dir", nargs="*", default=[], help="collection dirs whose *.json metadata provide native anchors")
    ap.add_argument("--perturbed", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clean_rows = load_rows(args.clean)
    pert_rows = load_rows(args.perturbed)
    native_success = {
        (r["suite"], r["task_idx"], r["episode_idx"]): r["success"] for r in clean_rows
    }
    for d in args.clean_collect_dir:
        for p in Path(d).glob("*.json"):
            if p.name.startswith(("collect_summary", "collection_")):
                continue
            m = json.loads(p.read_text())
            key = (str(m.get("suite")), int(m.get("task_idx", -1)), int(m.get("episode_idx", -1)))
            native_success[key] = bool(m.get("episode_success", False))

    certs, near_misses = [], 0
    for r in pert_rows:
        key = (r["suite"], r["task_idx"], r["episode_idx"])
        if not native_success.get(key, False):
            continue  # no native-success anchor -> not a counterfactual
        if r["success"]:
            near_misses += 1
            continue
        certs.append(
            {
                "name": r["perturbation"]["name"],
                "type": r["perturbation"]["type"],
                "params": r["perturbation"]["params"],
                "suite": r["suite"],
                "task_idx": r["task_idx"],
                "episode_idx": r["episode_idx"],
                "init_state_index": r.get("init_state_index"),
                "instruction": r.get("instruction"),
                "native_success": True,
                "perturbed_success": False,
            }
        )

    out = {
        "certificates": certs,
        "num_certificates": len(certs),
        "num_perturbed_rows": len(pert_rows),
        "num_native_anchored_survivals": near_misses,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "certificates"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
