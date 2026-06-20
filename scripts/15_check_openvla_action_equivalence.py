"""Compare SafeSAE's OpenVLA action path against the model's official predict_action API."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.rollout_collector import RolloutCollector  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.utils.runtime import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/rollout_success_debug.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True, choices=["spatial", "object", "goal", "long"])
    parser.add_argument("--task_idx", type=int, required=True)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument(
        "--advance_with",
        type=str,
        default="official",
        choices=["official", "equivalent", "legacy"],
        help="Which action path to use when stepping the env during the comparison rollout.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(diff)), float(np.max(np.abs(diff)))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    cfg = load_yaml(args.config)
    collector = RolloutCollector(cfg)

    checkpoint = args.checkpoint or collector.model_cfg["checkpoints"].get(args.suite, collector.model_name)
    collector._load_model(checkpoint)

    suite_obj = collector._suite_builder(args.suite)
    bddl = collector._task_bddl(suite_obj, args.task_idx)
    instruction = collector._task_instruction(suite_obj, args.task_idx)
    task_name = collector._task_name(suite_obj, args.task_idx, bddl)
    image_key = f"{collector.sim_cfg.get('camera_names', ['agentview'])[0]}_image"

    rows: list[dict[str, object]] = []
    reward_sum = 0.0
    done_count = 0

    env = collector._build_env(bddl)
    try:
        obs = env.reset()
        for step_idx in range(int(args.num_steps)):
            if image_key not in obs:
                raise KeyError(f"Observation missing camera image key: {image_key}")

            raw_inputs = collector._prepare_inputs(instruction, obs[image_key], apply_prompt_template=False)
            official_inputs = collector._prepare_inputs(instruction, obs[image_key], apply_prompt_template=True)
            raw_input_ids = raw_inputs["input_ids"].detach().cpu().numpy()
            official_input_ids = official_inputs["input_ids"].detach().cpu().numpy()

            legacy_action, legacy_tokens = collector._predict_action_custom(
                raw_inputs,
                suite=args.suite,
                append_prompt_token=False,
            )
            raw_plus_suffix_action, raw_plus_suffix_tokens = collector._predict_action_custom(
                raw_inputs,
                suite=args.suite,
                append_prompt_token=True,
            )
            official_prompt_equivalent_action, official_prompt_equivalent_tokens = collector._predict_action_custom(
                official_inputs,
                suite=args.suite,
                append_prompt_token=True,
            )
            official_action = collector._predict_action_official(official_inputs, suite=args.suite)

            legacy_l2, legacy_max_abs = _metrics(legacy_action, official_action)
            raw_plus_suffix_l2, raw_plus_suffix_max_abs = _metrics(raw_plus_suffix_action, official_action)
            official_prompt_equivalent_l2, official_prompt_equivalent_max_abs = _metrics(
                official_prompt_equivalent_action,
                official_action,
            )

            row = {
                "step": step_idx,
                "suite": args.suite,
                "task_idx": args.task_idx,
                "task_name": task_name,
                "instruction": instruction,
                "raw_prompt_last_token": int(raw_input_ids[0, -1]),
                "raw_prompt_len": int(raw_input_ids.shape[1]),
                "official_prompt_last_token": int(official_input_ids[0, -1]),
                "official_prompt_len": int(official_input_ids.shape[1]),
                "legacy_action_l2_vs_official": legacy_l2,
                "legacy_action_max_abs_vs_official": legacy_max_abs,
                "raw_plus_suffix_action_l2_vs_official": raw_plus_suffix_l2,
                "raw_plus_suffix_action_max_abs_vs_official": raw_plus_suffix_max_abs,
                "official_prompt_equivalent_action_l2_vs_official": official_prompt_equivalent_l2,
                "official_prompt_equivalent_action_max_abs_vs_official": official_prompt_equivalent_max_abs,
                "legacy_matches_official": bool(np.allclose(legacy_action, official_action, atol=1e-6)),
                "raw_plus_suffix_matches_official": bool(np.allclose(raw_plus_suffix_action, official_action, atol=1e-6)),
                "official_prompt_equivalent_matches_official": bool(
                    np.allclose(official_prompt_equivalent_action, official_action, atol=1e-6)
                ),
                "legacy_token_ids": " ".join(str(int(x)) for x in legacy_tokens.tolist()),
                "raw_plus_suffix_token_ids": " ".join(str(int(x)) for x in raw_plus_suffix_tokens.tolist()),
                "official_prompt_equivalent_token_ids": " ".join(
                    str(int(x)) for x in official_prompt_equivalent_tokens.tolist()
                ),
                "legacy_action": " ".join(f"{float(x):.8f}" for x in legacy_action.tolist()),
                "raw_plus_suffix_action": " ".join(f"{float(x):.8f}" for x in raw_plus_suffix_action.tolist()),
                "official_prompt_equivalent_action": " ".join(
                    f"{float(x):.8f}" for x in official_prompt_equivalent_action.tolist()
                ),
                "official_action": " ".join(f"{float(x):.8f}" for x in official_action.tolist()),
            }
            rows.append(row)

            if args.advance_with == "legacy":
                action_to_apply = legacy_action
            elif args.advance_with == "equivalent":
                action_to_apply = official_prompt_equivalent_action
            else:
                action_to_apply = official_action

            obs, reward, done, _info = env.step(action_to_apply)
            reward_sum += float(reward)
            done_count += int(bool(done))
            if done:
                break
    finally:
        env.close()

    _write_csv(output_dir / "per_step_action_equivalence.csv", rows)

    if rows:
        legacy_l2 = [float(row["legacy_action_l2_vs_official"]) for row in rows]
        legacy_max_abs = [float(row["legacy_action_max_abs_vs_official"]) for row in rows]
        raw_plus_suffix_l2 = [float(row["raw_plus_suffix_action_l2_vs_official"]) for row in rows]
        raw_plus_suffix_max_abs = [float(row["raw_plus_suffix_action_max_abs_vs_official"]) for row in rows]
        official_prompt_equivalent_l2 = [
            float(row["official_prompt_equivalent_action_l2_vs_official"]) for row in rows
        ]
        official_prompt_equivalent_max_abs = [
            float(row["official_prompt_equivalent_action_max_abs_vs_official"]) for row in rows
        ]
        summary = {
            "config": str(args.config),
            "checkpoint": checkpoint,
            "suite": args.suite,
            "task_idx": int(args.task_idx),
            "task_name": task_name,
            "instruction": instruction,
            "num_steps_requested": int(args.num_steps),
            "num_steps_run": len(rows),
            "advance_with": args.advance_with,
            "reward_sum": reward_sum,
            "done_count": done_count,
            "raw_processor_prompts_already_end_with_empty_token": all(
                int(row["raw_prompt_last_token"]) == collector.action_empty_token_id for row in rows
            ),
            "official_processor_prompts_already_end_with_empty_token": all(
                int(row["official_prompt_last_token"]) == collector.action_empty_token_id for row in rows
            ),
            "legacy_matches_official_all_steps": all(bool(row["legacy_matches_official"]) for row in rows),
            "raw_plus_suffix_matches_official_all_steps": all(
                bool(row["raw_plus_suffix_matches_official"]) for row in rows
            ),
            "official_prompt_equivalent_matches_official_all_steps": all(
                bool(row["official_prompt_equivalent_matches_official"]) for row in rows
            ),
            "legacy_mean_l2_vs_official": float(np.mean(legacy_l2)),
            "legacy_max_l2_vs_official": float(np.max(legacy_l2)),
            "legacy_mean_max_abs_vs_official": float(np.mean(legacy_max_abs)),
            "legacy_max_abs_vs_official": float(np.max(legacy_max_abs)),
            "raw_plus_suffix_mean_l2_vs_official": float(np.mean(raw_plus_suffix_l2)),
            "raw_plus_suffix_max_l2_vs_official": float(np.max(raw_plus_suffix_l2)),
            "raw_plus_suffix_mean_max_abs_vs_official": float(np.mean(raw_plus_suffix_max_abs)),
            "raw_plus_suffix_max_abs_vs_official": float(np.max(raw_plus_suffix_max_abs)),
            "official_prompt_equivalent_mean_l2_vs_official": float(np.mean(official_prompt_equivalent_l2)),
            "official_prompt_equivalent_max_l2_vs_official": float(np.max(official_prompt_equivalent_l2)),
            "official_prompt_equivalent_mean_max_abs_vs_official": float(np.mean(official_prompt_equivalent_max_abs)),
            "official_prompt_equivalent_max_abs_vs_official": float(np.max(official_prompt_equivalent_max_abs)),
        }
    else:
        summary = {
            "config": str(args.config),
            "checkpoint": checkpoint,
            "suite": args.suite,
            "task_idx": int(args.task_idx),
            "num_steps_requested": int(args.num_steps),
            "num_steps_run": 0,
            "advance_with": args.advance_with,
        }
    save_json(output_dir / "action_equivalence_summary.json", summary)


if __name__ == "__main__":
    main()
