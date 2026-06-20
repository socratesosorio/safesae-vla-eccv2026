"""Build a deterministic flat rollout subset from nested rollout chunks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RolloutRecord:
    source_json: str
    source_tensor: str
    suite: str
    task_idx: int
    noise_applied: bool
    noise_level: float
    total_violations: int
    num_steps: int

    @property
    def task_key(self) -> tuple[str, int]:
        return (self.suite, self.task_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        type=str,
        default="safesae_rollouts_from_modal/rollouts",
        help="Root directory containing rollout chunk subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Flat output directory that will contain rollout_*.json/.safetensors pairs.",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=2,
        help="Maximum number of episodes to keep per (suite, task_idx).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Optional cap on total selected episodes. Zero means no cap.",
    )
    parser.add_argument(
        "--link_mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to materialize tensor files in the flat subset directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    return parser.parse_args()


def discover_rollouts(source_dir: str) -> list[RolloutRecord]:
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    records: list[RolloutRecord] = []
    for meta_path in sorted(root.rglob("rollout_*.json")):
        tensor_path = meta_path.with_suffix(".safetensors")
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing tensor pair for {meta_path}")

        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        suite = meta.get("suite")
        task_idx = meta.get("task_idx")
        if suite is None or task_idx is None:
            continue

        records.append(
            RolloutRecord(
                source_json=str(meta_path.resolve()),
                source_tensor=str(tensor_path.resolve()),
                suite=str(suite),
                task_idx=int(task_idx),
                noise_applied=bool(meta.get("noise_applied", False)),
                noise_level=float(meta.get("noise_level", 0.0)),
                total_violations=int(meta.get("total_violations", 0)),
                num_steps=int(meta.get("num_steps", 0)),
            )
        )

    if not records:
        raise FileNotFoundError(f"No rollout metadata found under {root}")
    return records


def _diverse_task_order(records: Iterable[RolloutRecord]) -> list[RolloutRecord]:
    ordered = sorted(
        records,
        key=lambda rec: (
            rec.noise_level,
            int(rec.noise_applied),
            rec.total_violations,
            rec.source_json,
        ),
    )
    dq = deque(ordered)
    selection: list[RolloutRecord] = []
    take_low = True
    while dq:
        selection.append(dq.popleft() if take_low else dq.pop())
        take_low = not take_low
    return selection


def select_task_balanced_subset(
    records: list[RolloutRecord],
    episodes_per_task: int,
    max_episodes: int = 0,
) -> list[RolloutRecord]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")

    per_task: dict[tuple[str, int], list[RolloutRecord]] = defaultdict(list)
    for record in records:
        per_task[record.task_key].append(record)

    ordered_per_task = {task: _diverse_task_order(task_records) for task, task_records in per_task.items()}
    task_keys = sorted(ordered_per_task)
    selected: list[RolloutRecord] = []
    limit = max_episodes if max_episodes > 0 else None

    for round_idx in range(episodes_per_task):
        for task_key in task_keys:
            task_records = ordered_per_task[task_key]
            if round_idx >= len(task_records):
                continue
            selected.append(task_records[round_idx])
            if limit is not None and len(selected) >= limit:
                return selected

    return selected


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_tensor(src: Path, dst: Path, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    elif link_mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported link_mode: {link_mode}")


def summarize_selection(selected: list[RolloutRecord]) -> dict:
    by_suite = Counter(record.suite for record in selected)
    by_noise = Counter(f"{record.noise_level:.2f}" for record in selected)
    by_task = Counter(record.task_key for record in selected)
    total_bytes = sum(
        Path(record.source_json).stat().st_size + Path(record.source_tensor).stat().st_size for record in selected
    )
    return {
        "num_episodes": len(selected),
        "num_tasks": len(by_task),
        "episodes_per_task_min": min(by_task.values()) if by_task else 0,
        "episodes_per_task_max": max(by_task.values()) if by_task else 0,
        "suite_counts": dict(sorted(by_suite.items())),
        "noise_level_counts": dict(sorted(by_noise.items())),
        "estimated_total_bytes": total_bytes,
        "estimated_total_gb": total_bytes / (1024**3),
    }


def materialize_subset(selected: list[RolloutRecord], output_dir: str, link_mode: str, overwrite: bool) -> dict:
    out_dir = Path(output_dir)
    _prepare_output_dir(out_dir, overwrite=overwrite)

    manifest_entries = []
    for new_idx, record in enumerate(selected):
        target_stem = f"rollout_{new_idx:06d}"
        target_json = out_dir / f"{target_stem}.json"
        target_tensor = out_dir / f"{target_stem}.safetensors"

        with open(record.source_json, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["subset_target_stem"] = target_stem
        meta["subset_source_json"] = record.source_json
        meta["subset_source_tensor"] = record.source_tensor
        meta["subset_source_task_key"] = {"suite": record.suite, "task_idx": record.task_idx}
        meta["subset_source_noise_level"] = record.noise_level
        meta["subset_source_noise_applied"] = record.noise_applied

        with target_json.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

        _materialize_tensor(Path(record.source_tensor), target_tensor, link_mode=link_mode)

        manifest_entries.append(
            {
                "target_stem": target_stem,
                "target_json": str(target_json.resolve()),
                "target_tensor": str(target_tensor.resolve()),
                **asdict(record),
            }
        )

    summary = summarize_selection(selected)
    summary["link_mode"] = link_mode

    manifest = {
        "selection_strategy": "task_balanced_low_high_round_robin",
        "episodes_per_task": max(Counter(record.task_key for record in selected).values(), default=0),
        "summary": summary,
        "entries": manifest_entries,
    }

    manifest_path = out_dir / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    summary_path = out_dir / "subset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return manifest


def main() -> None:
    args = parse_args()
    records = discover_rollouts(args.source_dir)
    selected = select_task_balanced_subset(
        records,
        episodes_per_task=args.episodes_per_task,
        max_episodes=args.max_episodes,
    )
    manifest = materialize_subset(
        selected,
        output_dir=args.output_dir,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
