"""Aggregate RL4VLA loop results into the numbers used by the three papers.

Run on the cluster (or locally on synced results):
  python analyze_results.py --root /work/joy/rl4vla_loops
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def rows(pattern: str) -> list[dict]:
    out = []
    for p in glob.glob(pattern):
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def succ(rs: list[dict]) -> tuple[int, int]:
    return sum(1 for r in rs if r["success"]), len(rs)


def latest_val(metrics_path: Path) -> dict | None:
    last = None
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("split") == "val":
                last = row
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/joy/rl4vla_loops")
    args = ap.parse_args()
    root = Path(args.root)
    report: dict = {}

    # ---- collection stats ----
    metas = [json.loads(p.read_text()) for p in (root / "data/object_train").glob("*.json")
             if not p.name.startswith(("collect_summary", "collection_"))]
    n_succ = sum(1 for m in metas if m.get("episode_success"))
    report["collection"] = {"episodes": len(metas), "successes": n_succ}

    # ---- clean evals per arm ----
    clean = {}
    base_rows = rows(str(root / "evals/base/base_clean_*.jsonl"))
    clean["base"] = succ(base_rows)
    for arm_dir in sorted((root / "evals").glob("clean_*")):
        arm = arm_dir.name.replace("clean_", "")
        rs = rows(str(arm_dir / "*.jsonl"))
        clean[arm] = succ(rs)
    report["clean_eval"] = {k: {"successes": v[0], "n": v[1]} for k, v in clean.items()}

    # ---- per-arm val metrics (CE / action-MSE trajectories) ----
    val = {}
    for run_dir in sorted((root / "runs").glob("*")):
        m = latest_val(run_dir / "metrics.jsonl")
        if m:
            val[run_dir.name] = {"val_ce": m["ce"], "val_action_mse": m["action_mse"], "step": m["step"]}
    report["val_metrics"] = val

    # ---- certificates ----
    for name in ("certs_train", "certs_holdout"):
        p = root / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            by_fam = defaultdict(int)
            for c in d["certificates"]:
                by_fam[c["type"]] += 1
            report[name] = {"n": d["num_certificates"], "by_family": dict(by_fam)}

    # ---- closure evals: evals/closure_<arm>/*.jsonl rows replay held-out certs ----
    closure = {}
    for arm_dir in sorted((root / "evals").glob("closure_*")):
        arm = arm_dir.name.replace("closure_", "")
        rs = rows(str(arm_dir / "*.jsonl"))
        # closure = fraction of certified (previously failing) replays that now succeed
        k, n = succ(rs)
        closure[arm] = {"closed": k, "n": n, "closure_rate": (k / n) if n else None}
    # base closure from the held-out mining rows themselves (those that FAILED define certs;
    # base closure is 0 by construction on exact certs)
    report["closure"] = closure

    # ---- neighborhood closure: evals/nbhd_<arm>/ ----
    nbhd = {}
    for arm_dir in sorted((root / "evals").glob("nbhd_*")):
        arm = arm_dir.name.replace("nbhd_", "")
        rs = rows(str(arm_dir / "*.jsonl"))
        k, n = succ(rs)
        nbhd[arm] = {"closed": k, "n": n}
    if nbhd:
        report["neighborhood_closure"] = nbhd

    # ---- phi diagnostics ----
    for run_dir in sorted((root / "runs").glob("*phi*")):
        p = run_dir / "phi_weights.json"
        if p.exists():
            d = json.loads(p.read_text())
            report["phi"] = {"episode_phi_auroc": d.get("episode_phi_auroc"), "tau": d.get("tau")}

    print(json.dumps(report, indent=1))
    (root / "analysis_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
