import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.causal_validation import (
    _compute_action_effect_metrics,
    _aggregate_metrics,
    _rollout_metrics,
    load_feature_sets,
    result_to_summary_row,
)
from src.utils.hooks import apply_feature_intervention


def test_apply_feature_intervention_supports_scale_map_and_fallback():
    features = torch.ones((1, 6), dtype=torch.float32)
    modified = apply_feature_intervention(
        features=features,
        feature_indices=[1, 2, 3],
        scale=0.5,
        feature_scale_map={1: 0.0, 3: 2.0},
    )
    expected = torch.tensor([[1.0, 0.0, 0.5, 2.0, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(modified, expected)


def test_targeted_causal_sweep_manifest_and_loader(tmp_path: Path):
    monitor_df = pd.DataFrame(
        [
            {
                "feature_idx": 101,
                "consensus_rank": 1,
                "topk_frequency": 0.90,
                "positive_weight_fraction": 0.95,
                "mean_signed_weight": 0.80,
                "mean_normalized_abs_weight": 0.80,
                "mean_abs_weight": 0.80,
            },
            {
                "feature_idx": 102,
                "consensus_rank": 2,
                "topk_frequency": 0.80,
                "positive_weight_fraction": 0.85,
                "mean_signed_weight": 0.60,
                "mean_normalized_abs_weight": 0.70,
                "mean_abs_weight": 0.70,
            },
            {
                "feature_idx": 103,
                "consensus_rank": 3,
                "topk_frequency": 0.75,
                "positive_weight_fraction": 0.10,
                "mean_signed_weight": -0.55,
                "mean_normalized_abs_weight": 0.60,
                "mean_abs_weight": 0.60,
            },
            {
                "feature_idx": 201,
                "consensus_rank": 4,
                "topk_frequency": 0.88,
                "positive_weight_fraction": 0.92,
                "mean_signed_weight": 0.72,
                "mean_normalized_abs_weight": 0.77,
                "mean_abs_weight": 0.77,
            },
            {
                "feature_idx": 202,
                "consensus_rank": 5,
                "topk_frequency": 0.70,
                "positive_weight_fraction": 0.08,
                "mean_signed_weight": -0.50,
                "mean_normalized_abs_weight": 0.52,
                "mean_abs_weight": 0.52,
            },
        ]
    )
    monitor_csv = tmp_path / "layer16_sae_feature_weights.csv"
    monitor_df.to_csv(monitor_csv, index=False)

    category_dir = tmp_path / "category_features"
    category_dir.mkdir()
    pd.DataFrame(
        [
            {"feature_idx": 101, "composite_score": 9.0, "direction": "higher_in_unsafe"},
            {"feature_idx": 102, "composite_score": 8.0, "direction": "higher_in_unsafe"},
            {"feature_idx": 103, "composite_score": 7.5, "direction": "higher_in_safe"},
        ]
    ).to_csv(category_dir / "layer16_collision.csv", index=False)
    pd.DataFrame(
        [
            {"feature_idx": 201, "composite_score": 8.5, "direction": "higher_in_unsafe"},
            {"feature_idx": 202, "composite_score": 7.0, "direction": "higher_in_safe"},
        ]
    ).to_csv(category_dir / "layer16_boundary_violation.csv", index=False)

    out_dir = tmp_path / "out"
    screened_cells_csv = tmp_path / "screened_cells.csv"
    pd.DataFrame(
        [
            {
                "target_category": "collision",
                "suite": "goal",
                "task_idx": 0,
                "condition": "hazard_collision",
                "num_episodes": 6,
                "success_rate": 0.5,
                "clean_success_rate": 0.2,
                "target_category_rate": 0.5,
                "recommended": True,
            },
            {
                "target_category": "boundary_violation",
                "suite": "spatial",
                "task_idx": 2,
                "condition": "hazard_boundary_violation",
                "num_episodes": 6,
                "success_rate": 0.4,
                "clean_success_rate": 0.2,
                "target_category_rate": 0.6,
                "recommended": True,
            },
        ]
    ).to_csv(screened_cells_csv, index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/18_prepare_targeted_causal_sweeps.py",
            "--feature_weights_csv",
            str(monitor_csv),
            "--category_feature_dir",
            str(category_dir),
            "--categories",
            "collision,boundary_violation",
            "--max_risky_single_features",
            "2",
            "--max_protective_single_features",
            "1",
            "--risky_pair_pool_size",
            "2",
            "--protective_pair_pool_size",
            "1",
            "--output_dir",
            str(out_dir),
            "--output_prefix",
            "pilot",
            "--screened_cells_csv",
            str(screened_cells_csv),
            "--suppress_scales",
            "0.0,0.5",
            "--boost_scales",
            "1.25",
        ],
        check=True,
    )

    manifest_csv = out_dir / "pilot_manifest.csv"
    assert manifest_csv.exists()

    manifest_df = pd.read_csv(manifest_csv)
    assert {"feature_set", "feature_idx", "feature_scale", "target_category", "hazard_category", "condition_group", "intervention_direction", "selection_strategy"}.issubset(
        set(manifest_df.columns)
    )
    assert "collision_risky_single_f101_sc0" in set(manifest_df["feature_set"])
    assert "collision_risky_single_f101_sc0p5" in set(manifest_df["feature_set"])
    assert "collision_risky_pair_f101_f102_sc0p5" in set(manifest_df["feature_set"])
    assert "boundary_violation_risky_single_f201_sc0p5" in set(manifest_df["feature_set"])
    assert "collision_protective_single_f103_sc1p25" in set(manifest_df["feature_set"])
    assert set(manifest_df["condition_group"]) == {"hazard_targeted"}
    assert {"allowed_task_specs", "trigger_mode", "trigger_threshold", "trigger_start_step", "trigger_latch"}.issubset(set(manifest_df.columns))

    feature_sets = load_feature_sets(
        features_csv="",
        top_k=5,
        scale=0.0,
        random_controls_csv="",
        feature_manifest_csv=str(manifest_csv),
    )
    names = {spec["feature_set"] for spec in feature_sets}
    assert "collision_risky_single_f101_sc0p5" in names
    risky_single = next(spec for spec in feature_sets if spec["feature_set"] == "collision_risky_single_f101_sc0p5")
    assert risky_single["target_category"] == "collision"
    assert risky_single["hazard_category"] == "collision"
    assert risky_single["feature_scale_map"] == {101: 0.5}
    assert risky_single["allowed_task_specs"] == ["goal:0"]
    assert risky_single["condition_names"] == ["hazard_collision"]
    assert risky_single["trigger_mode"] == "wrist_force_ratio"
    risky_pair = next(spec for spec in feature_sets if spec["feature_set"] == "collision_risky_pair_f101_f102_sc0p5")
    assert risky_pair["feature_scale_map"] == {101: 0.5, 102: 0.5}
    boundary_single = next(spec for spec in feature_sets if spec["feature_set"] == "boundary_violation_risky_single_f201_sc0p5")
    assert boundary_single["allowed_task_specs"] == ["spatial:2"]
    assert boundary_single["trigger_mode"] == "boundary_margin"


def test_always_on_sanity_manifest_builder(tmp_path: Path):
    manifest_csv = tmp_path / "source_manifest.csv"
    pd.DataFrame(
        [
            {
                "feature_set": "excessive_force_risky_single_f11230_sc0",
                "rank": 1,
                "feature_idx": 11230,
                "feature_scale": 0.0,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "suppress",
                "selection_strategy": "category_risky_single",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            },
            {
                "feature_set": "excessive_force_risky_single_f2332_sc0p25",
                "rank": 1,
                "feature_idx": 2332,
                "feature_scale": 0.25,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "suppress",
                "selection_strategy": "category_risky_single",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            },
            {
                "feature_set": "excessive_force_risky_pair_f11230_f2332_sc0p5",
                "rank": 1,
                "feature_idx": 11230,
                "feature_scale": 0.5,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "suppress",
                "selection_strategy": "category_risky_pair",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            },
            {
                "feature_set": "excessive_force_risky_pair_f11230_f2332_sc0p5",
                "rank": 2,
                "feature_idx": 2332,
                "feature_scale": 0.5,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "suppress",
                "selection_strategy": "category_risky_pair",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            },
        ]
    ).to_csv(manifest_csv, index=False)

    out_dir = tmp_path / "always_on"
    subprocess.run(
        [
            sys.executable,
            "scripts/20_prepare_always_on_sanity_sweeps.py",
            "--feature_manifest_csv",
            str(manifest_csv),
            "--output_dir",
            str(out_dir),
            "--output_prefix",
            "pilot_always_on",
            "--target_category",
            "excessive_force",
            "--intervention_direction",
            "suppress",
            "--max_feature_sets",
            "2",
            "--prefer_single_features",
            "1",
            "--recommended_num_rollouts",
            "3",
        ],
        check=True,
    )

    always_manifest_csv = out_dir / "pilot_always_on_manifest.csv"
    always_summary_json = out_dir / "pilot_always_on_summary.json"
    assert always_manifest_csv.exists()
    assert always_summary_json.exists()

    always_df = pd.read_csv(always_manifest_csv)
    assert set(always_df["feature_set"]) == {
        "excessive_force_risky_single_f11230_sc0_always_on",
        "excessive_force_risky_single_f2332_sc0p25_always_on",
    }
    assert set(always_df["trigger_mode"]) == {"always"}
    assert always_df["trigger_threshold"].isna().all()
    assert set(always_df["trigger_start_step"]) == {0}
    assert set(always_df["sanity_mode"]) == {"always_on"}
    assert set(always_df["sanity_source_feature_set"]) == {
        "excessive_force_risky_single_f11230_sc0",
        "excessive_force_risky_single_f2332_sc0p25",
    }

    feature_sets = load_feature_sets(
        features_csv="",
        top_k=5,
        scale=0.0,
        random_controls_csv="",
        feature_manifest_csv=str(always_manifest_csv),
    )
    names = {spec["feature_set"] for spec in feature_sets}
    assert "excessive_force_risky_single_f11230_sc0_always_on" in names
    spec = next(spec for spec in feature_sets if spec["feature_set"] == "excessive_force_risky_single_f11230_sc0_always_on")
    assert spec["trigger_mode"] == "always"
    assert spec["allowed_task_specs"] == ["goal:0"]


def test_refined_followup_builder_generates_gentler_triggered_scales(tmp_path: Path):
    controls_csv = tmp_path / "controls.csv"
    pd.DataFrame(
        [
            {
                "feature_set": "excessive_force_protective_single_f7587_sc1p25_always_on",
                "is_random_control": 0,
                "feature_scale_mean": 1.25,
                "target_category": "excessive_force",
                "intervention_direction": "boost",
                "success_rate_clamped": 0.0,
                "success_rate_baseline": 0.75,
                "any_violation_rate_clamped": 0.75,
                "any_violation_rate_baseline": 1.0,
                "target_category_rate_clamped": 0.75,
                "target_category_rate_baseline": 1.0,
                "action_delta_any_nonzero_rate_clamped": 1.0,
            },
            {
                "feature_set": "excessive_force_risky_single_f11230_sc0_always_on",
                "is_random_control": 0,
                "feature_scale_mean": 0.0,
                "target_category": "excessive_force",
                "intervention_direction": "suppress",
                "success_rate_clamped": 0.25,
                "success_rate_baseline": 0.75,
                "any_violation_rate_clamped": 1.0,
                "any_violation_rate_baseline": 1.0,
                "target_category_rate_clamped": 1.0,
                "target_category_rate_baseline": 1.0,
                "action_delta_any_nonzero_rate_clamped": 0.5,
            },
        ]
    ).to_csv(controls_csv, index=False)

    source_manifest_csv = tmp_path / "source_manifest.csv"
    pd.DataFrame(
        [
            {
                "feature_set": "excessive_force_protective_single_f7587_sc1p25",
                "rank": 1,
                "feature_idx": 7587,
                "feature_scale": 1.25,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "boost",
                "selection_strategy": "category_protective_single",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            }
        ]
    ).to_csv(source_manifest_csv, index=False)

    out_dir = tmp_path / "refined"
    subprocess.run(
        [
            sys.executable,
            "scripts/21_prepare_refined_causal_followups.py",
            "--controls_csv",
            str(controls_csv),
            "--source_manifest_csv",
            str(source_manifest_csv),
            "--output_dir",
            str(out_dir),
            "--output_prefix",
            "followup",
            "--target_category",
            "excessive_force",
            "--max_candidates",
            "1",
            "--max_success_drop",
            "0.25",
            "--min_target_improvement",
            "0.10",
            "--gentle_boost_scales",
            "1.02,1.05,1.1,1.15,1.2",
        ],
        check=True,
    )

    manifest_csv = out_dir / "followup_manifest.csv"
    summary_json = out_dir / "followup_summary.json"
    assert manifest_csv.exists()
    assert summary_json.exists()

    manifest_df = pd.read_csv(manifest_csv)
    assert set(manifest_df["feature_set"]) == {
        "excessive_force_protective_single_f7587_sc1p25_refine_sc1p02",
        "excessive_force_protective_single_f7587_sc1p25_refine_sc1p05",
        "excessive_force_protective_single_f7587_sc1p25_refine_sc1p1",
        "excessive_force_protective_single_f7587_sc1p25_refine_sc1p15",
        "excessive_force_protective_single_f7587_sc1p25_refine_sc1p2",
    }
    assert set(manifest_df["trigger_mode"]) == {"wrist_force_ratio"}
    assert set(manifest_df["trigger_start_step"]) == {10}
    assert set(manifest_df["feature_scale"]) == {1.02, 1.05, 1.1, 1.15, 1.2}
    assert set(manifest_df["refine_selection_mode"]) == {"fallback_positive_but_too_harmful"}
    assert set(manifest_df["refine_parent_feature_set"]) == {
        "excessive_force_protective_single_f7587_sc1p25_always_on"
    }

    summary = json.loads(summary_json.read_text())
    assert summary["recommended_num_rollouts"] == 6


def test_causal_telemetry_propagates_into_summary_rows():
    baseline_rollout = {
        "actions": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "safety_labels": torch.zeros((3, 5), dtype=torch.bool).numpy(),
        "episode_success": True,
        "metadata": {
            "suite": "goal",
            "task_idx": 0,
            "collection_condition": "hazard_excessive_force",
            "collection_condition_group": "hazard_targeted",
            "hazard_category": "excessive_force",
        },
    }
    clamped_rollout = {
        "actions": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            ],
            dtype=np.float32,
        ),
        "safety_labels": torch.tensor(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        ).numpy(),
        "episode_success": True,
        "metadata": {
            "suite": "goal",
            "task_idx": 0,
            "collection_condition": "hazard_excessive_force",
            "collection_condition_group": "hazard_targeted",
            "hazard_category": "excessive_force",
            "intervention_enabled": True,
            "intervention_triggered": True,
            "intervention_trigger_mode": "wrist_force_ratio",
            "intervention_trigger_signal_name": "wrist_force_ratio",
            "intervention_first_trigger_step": 2,
            "intervention_first_active_step": 2,
            "intervention_active_steps": 5,
            "intervention_trigger_eval_count": 8,
            "intervention_trigger_true_count": 3,
            "intervention_trigger_true_fraction": 0.375,
            "intervention_active_fraction": 0.625,
            "intervention_mean_trigger_value": 1.2,
            "intervention_mean_trigger_margin": 0.2,
            "intervention_max_trigger_value": 1.8,
            "intervention_max_trigger_margin": 0.8,
            "intervention_active_by_timestep": [False, True, True],
        },
    }

    clamped_row = _rollout_metrics(clamped_rollout)
    clamped_row.update(_compute_action_effect_metrics(baseline_rollout, clamped_rollout))
    baseline_row = _rollout_metrics(baseline_rollout)
    assert clamped_row["intervention_triggered"] is True
    assert clamped_row["intervention_first_trigger_step"] == 2
    assert clamped_row["intervention_trigger_true_fraction"] == 0.375
    assert clamped_row["action_delta_any_nonzero"] is True
    assert clamped_row["action_delta_active_steps"] == 2

    result = {
        "clamped": _aggregate_metrics([clamped_row]),
        "baseline": _aggregate_metrics([baseline_row]),
        "paired_test": {
            "collision_wilcoxon_p": 1.0,
            "any_violation_wilcoxon_p": 1.0,
            "success_wilcoxon_p": 1.0,
            "clean_success_wilcoxon_p": 1.0,
            "target_category_wilcoxon_p": 1.0,
        },
        "config": {
            "target_category": "excessive_force",
            "hazard_category": "excessive_force",
            "condition_group": "hazard_targeted",
            "condition_names": ["hazard_excessive_force"],
            "allowed_suites": ["goal"],
            "allowed_task_specs": ["goal:0"],
            "selection_strategy": "signed_single",
            "intervention_direction": "suppress",
            "sampling_pool_size": 1,
            "trigger_mode": "wrist_force_ratio",
            "trigger_threshold": 0.5,
            "trigger_start_step": 10,
            "trigger_end_step": None,
            "trigger_latch": True,
            "feature_scale_map": {"123": 0.5},
        },
    }
    summary = result_to_summary_row(
        result=result,
        feature_set="excessive_force_risky_single_f123_sc0p5",
        is_random_control=False,
        scale=0.5,
        num_rollouts=1,
        num_features=1,
    )
    assert summary["intervention_trigger_signal_name"] == "wrist_force_ratio"
    assert summary["intervention_triggered_rate_clamped"] == 1.0
    assert summary["mean_first_trigger_step_clamped"] == 2.0
    assert summary["mean_trigger_true_fraction_clamped"] == 0.375
    assert summary["mean_trigger_value_clamped"] == 1.2
    assert summary["action_delta_any_nonzero_rate_clamped"] == 1.0
    assert summary["mean_action_delta_active_steps_clamped"] == 2.0
    assert summary["mean_action_delta_active_l2_clamped"] is not None


def test_trigger_policy_sweep_builder_generates_grid(tmp_path: Path):
    source_manifest_csv = tmp_path / "source_manifest.csv"
    pd.DataFrame(
        [
            {
                "feature_set": "excessive_force_protective_single_f7587_sc1p25",
                "rank": 1,
                "feature_idx": 7587,
                "feature_scale": 1.25,
                "target_category": "excessive_force",
                "hazard_category": "excessive_force",
                "condition_group": "hazard_targeted",
                "condition_names": "hazard_excessive_force",
                "intervention_direction": "boost",
                "selection_strategy": "category_protective_single",
                "allowed_suites": "goal",
                "allowed_task_specs": "goal:0",
                "trigger_mode": "wrist_force_ratio",
                "trigger_threshold": 0.5,
                "trigger_start_step": 10,
                "trigger_end_step": None,
                "trigger_latch": 1,
                "is_random_control": 0,
            }
        ]
    ).to_csv(source_manifest_csv, index=False)

    out_dir = tmp_path / "trigger_sweep"
    subprocess.run(
        [
            sys.executable,
            "scripts/22_prepare_trigger_policy_sweeps.py",
            "--source_manifest_csv",
            str(source_manifest_csv),
            "--output_dir",
            str(out_dir),
            "--output_prefix",
            "pilot_trigger_sweep",
            "--feature_idx",
            "7587",
            "--target_category",
            "excessive_force",
            "--scales",
            "1.05,1.10",
            "--trigger_thresholds",
            "0.05,0.10",
            "--trigger_start_steps",
            "0,5",
            "--trigger_latch_values",
            "1,0",
            "--include_always_on_anchors",
            "1",
            "--recommended_num_rollouts",
            "4",
        ],
        check=True,
    )

    manifest_csv = out_dir / "pilot_trigger_sweep_manifest.csv"
    summary_json = out_dir / "pilot_trigger_sweep_summary.json"
    assert manifest_csv.exists()
    assert summary_json.exists()

    manifest_df = pd.read_csv(manifest_csv)

    always_rows = manifest_df[manifest_df["sweep_group"] == "always_on_anchor"]
    triggered_rows = manifest_df[manifest_df["sweep_group"] == "trigger_gated"]

    assert len(always_rows) == 2
    assert set(always_rows["trigger_mode"]) == {"always"}
    assert set(always_rows["feature_scale"]) == {1.05, 1.10}

    assert len(triggered_rows) == 2 * 2 * 2 * 2
    assert set(triggered_rows["trigger_mode"]) == {"wrist_force_ratio"}
    assert set(triggered_rows["trigger_threshold"]) == {0.05, 0.10}
    assert set(triggered_rows["trigger_start_step"]) == {0, 5}
    assert set(triggered_rows["trigger_latch"]) == {0, 1}
    assert set(triggered_rows["feature_scale"]) == {1.05, 1.10}

    for col in ["allowed_suites", "allowed_task_specs", "target_category", "hazard_category", "condition_names"]:
        assert col in manifest_df.columns

    feature_sets = load_feature_sets(
        features_csv="",
        top_k=5,
        scale=0.0,
        random_controls_csv="",
        feature_manifest_csv=str(manifest_csv),
    )
    names = {spec["feature_set"] for spec in feature_sets}
    assert len(names) == len(always_rows) + len(triggered_rows)

    always_spec = next(
        spec
        for spec in feature_sets
        if spec["feature_set"].endswith("_always") and spec["feature_scale_map"].get(7587) == 1.05
    )
    assert always_spec["trigger_mode"] == "always"
    assert always_spec["allowed_task_specs"] == ["goal:0"]

    triggered_spec = next(
        spec
        for spec in feature_sets
        if "th0p05_s0_latch" in spec["feature_set"]
    )
    assert triggered_spec["trigger_mode"] == "wrist_force_ratio"
    assert triggered_spec["trigger_threshold"] == 0.05
    assert triggered_spec["trigger_start_step"] == 0
    assert triggered_spec["trigger_latch"] is True

    summary = json.loads(summary_json.read_text())
    assert summary["feature_idx"] == 7587
    assert summary["recommended_num_rollouts"] == 4
    assert summary["num_feature_sets"] == len(names)
