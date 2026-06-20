from random import Random

from src.data.collection_schedule import build_collection_schedule, summarize_collection_schedule


def test_build_collection_schedule_benchmark_counts():
    collection_cfg = {
        "schedule_mode": "benchmark",
        "shuffle_schedule": False,
        "per_suite": {"goal": 0, "object": 0},
        "benchmark": {
            "suites": ["goal", "object"],
            "task_selection": {
                "default_mode": "first_n",
                "default_first_n_tasks": 2,
            },
            "conditions": {
                "clean": {
                    "rollouts_per_task": 2,
                    "perturbation_type": "clean",
                    "perturbation_family": "none",
                    "perturbation_level": "none",
                    "noise_std": 0.0,
                },
                "mild": {
                    "rollouts_per_task": 1,
                    "perturbation_type": "mild_action_noise",
                    "perturbation_family": "action_noise",
                    "perturbation_level": "mild",
                    "noise_std": 0.03,
                },
                "hazard_collision": {
                    "rollouts_per_task": 1,
                    "condition_group": "hazard_targeted",
                    "hazard_category": "collision",
                },
            },
        },
    }

    schedule = build_collection_schedule(
        collection_cfg,
        suite_task_counts={"goal": 5, "object": 3},
        rng=Random(0),
    )
    summary = summarize_collection_schedule(schedule)

    assert len(schedule) == 16
    assert summary["suite_counts"] == {"goal": 8, "object": 8}
    assert summary["condition_counts"] == {"clean": 8, "hazard_collision": 4, "mild": 4}
    assert summary["tasks_per_suite"] == {"goal": 2, "object": 2}

    hazard_entries = [entry for entry in schedule if entry["condition"] == "hazard_collision"]
    assert len(hazard_entries) == 4
    assert all(entry["hazard_category"] == "collision" for entry in hazard_entries)
    assert all(entry["perturbation_family"] == "hazard_targeted" for entry in hazard_entries)
    assert all(float(entry["action_scale"]) > 1.0 for entry in hazard_entries)


def test_build_collection_schedule_legacy_counts():
    collection_cfg = {
        "schedule_mode": "legacy_noise",
        "shuffle_schedule": False,
        "per_suite": {"goal": 3},
        "noise": {
            "clean_fraction": 1.0,
            "mild_noise_fraction": 0.0,
            "strong_noise_fraction": 0.0,
            "mild_noise_std": 0.03,
            "strong_noise_std": 0.08,
        },
    }

    schedule = build_collection_schedule(
        collection_cfg,
        suite_task_counts={"goal": 2},
        rng=Random(0),
    )

    assert len(schedule) == 3
    assert [entry["task_idx"] for entry in schedule] == [0, 1, 0]
    assert all(entry["condition"] == "clean" for entry in schedule)
    assert all(float(entry["noise_level"]) == 0.0 for entry in schedule)
