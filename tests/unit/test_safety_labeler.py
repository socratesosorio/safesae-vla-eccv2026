from src.data.safety_labeler import SafetyLabeler, resolve_safety_config


def test_label_timestep_all_false_default():
    cfg = {"simulation": {"control_freq": 20}, "safety": {}}
    labeler = SafetyLabeler(cfg)
    out = labeler.label_timestep(
        {
            "contacts": [],
            "max_contact_force": 0.0,
            "eef_pos": [0.0, 0.0, 0.9],
            "eef_speed": 0.0,
        }
    )
    assert out == {
        "collision": False,
        "excessive_force": False,
        "boundary_violation": False,
        "high_approach_speed": False,
        "object_drop": False,
    }


def test_label_timestep_detects_force_and_collision():
    cfg = {"simulation": {"control_freq": 20}, "safety": {"excessive_force_threshold": 50.0}}
    labeler = SafetyLabeler(cfg)
    out = labeler.label_timestep(
        {
            "contacts": [{"force": 10.0, "expected": False}],
            "max_contact_force": 60.0,
            "eef_pos": [0.0, 0.0, 0.9],
            "eef_speed": 0.0,
        }
    )
    assert out["collision"] is True
    assert out["excessive_force"] is True


def test_label_episode_summary_counts():
    cfg = {"simulation": {"control_freq": 20}, "safety": {}}
    labeler = SafetyLabeler(cfg)
    ep = [
        {
            "contacts": [{"force": 2.0, "expected": False}],
            "max_contact_force": 2.0,
            "eef_pos": [0.0, 0.0, 0.9],
            "eef_speed": 0.0,
        },
        {
            "contacts": [],
            "max_contact_force": 0.0,
            "eef_pos": [1.0, 0.0, 0.9],
            "eef_speed": 0.0,
        },
    ]
    summary = labeler.label_episode(ep)
    assert summary["episode_unsafe"] is True
    assert summary["episode_violation_counts"]["collision"] >= 1
    assert summary["episode_violation_counts"]["boundary_violation"] >= 1


def test_resolve_safety_config_applies_per_suite_overrides():
    cfg = {
        "safety": {
            "collision_force_threshold": 5.0,
            "boundary_bounds": {"x": [-0.4, 0.4], "y": [-0.4, 0.4], "z": [0.78, 1.25]},
            "per_suite_overrides": {
                "object": {
                    "collision_force_threshold": 250.0,
                    "boundary_bounds": {"z": [0.0, 0.35]},
                }
            },
        }
    }

    resolved = resolve_safety_config(cfg, suite="object")
    assert resolved["collision_force_threshold"] == 250.0
    assert resolved["boundary_bounds"]["x"] == [-0.4, 0.4]
    assert resolved["boundary_bounds"]["z"] == [0.0, 0.35]


def test_safety_labeler_with_suite_uses_override_thresholds():
    cfg = {
        "simulation": {"control_freq": 20},
        "safety": {
            "collision_force_threshold": 5.0,
            "speed_threshold": 0.3,
            "boundary_bounds": {"x": [-0.4, 0.4], "y": [-0.4, 0.4], "z": [0.78, 1.25]},
            "per_suite_overrides": {
                "object": {
                    "collision_force_threshold": 250.0,
                    "speed_threshold": 1.2,
                    "boundary_bounds": {"x": [-0.3, 0.3], "y": [-0.6, 0.2], "z": [0.0, 0.35]},
                }
            },
        },
    }
    default_labeler = SafetyLabeler(cfg)
    object_labeler = default_labeler.with_suite("object")

    assert object_labeler.thresholds.collision_force_threshold == 250.0
    assert object_labeler.thresholds.speed_threshold == 1.2
    assert object_labeler.thresholds.boundary_z == (0.0, 0.35)

    out = object_labeler.label_timestep(
        {
            "contacts": [{"force": 10.0, "expected": False}],
            "max_contact_force": 10.0,
            "eef_pos": [0.0, 0.0, 0.2],
            "eef_speed": 0.4,
        }
    )
    assert out["collision"] is False
    assert out["boundary_violation"] is False
    assert out["high_approach_speed"] is False
