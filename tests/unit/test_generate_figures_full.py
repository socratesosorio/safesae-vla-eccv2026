import os
import subprocess
from pathlib import Path

import pandas as pd


def test_generate_figures_full_outputs(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    paper = tmp_path / "paper"

    (results / "differential").mkdir(parents=True, exist_ok=True)
    (results / "monitor").mkdir(parents=True, exist_ok=True)
    (results / "causal").mkdir(parents=True, exist_ok=True)
    (results / "ablations").mkdir(parents=True, exist_ok=True)
    (paper / "tables").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "feature_idx": [1, 2, 3, 4],
            "effect_size": [0.5, 0.2, -0.3, 0.1],
            "adjusted_p": [1e-4, 1e-2, 1e-3, 0.05],
            "significant": [True, True, True, False],
        }
    ).to_csv(results / "differential" / "layer16_overall.csv", index=False)

    pd.DataFrame(
        {
            "cat_a": ["collision", "collision"],
            "cat_b": ["collision", "object_drop"],
            "jaccard": [1.0, 0.2],
        }
    ).to_csv(results / "differential" / "layer16_category_overlap.csv", index=False)

    per_cat_effects = {
        "collision": 0.111,
        "excessive_force": -0.222,
        "boundary_violation": 0.333,
        "high_approach_speed": 0.444,
        "object_drop": -0.555,
    }
    per_cat_sig_counts = {
        "collision": [True, False],
        "excessive_force": [True, True],
        "boundary_violation": [False, False],
        "high_approach_speed": [True, False],
        "object_drop": [False, True],
    }
    for category, effect in per_cat_effects.items():
        pd.DataFrame(
            {
                "feature_idx": [1, 2],
                "effect_size": [effect, effect / 2.0],
                "significant": per_cat_sig_counts[category],
            }
        ).to_csv(results / "differential" / f"layer16_{category}.csv", index=False)

    pd.DataFrame(
        {
            "method": ["sae_lr", "force_threshold", "random"],
            "threshold": [0.5, 1.0, 0.5],
            "auroc": [0.8, 0.6, 0.5],
            "f1": [0.7, 0.4, 0.1],
            "precision": [0.8, 0.4, 0.1],
            "recall": [0.6, 0.5, 0.1],
            "cost_weighted_f1": [0.65, 0.45, 0.1],
            "pr_auc": [0.7, 0.5, 0.1],
        }
    ).to_csv(results / "monitor" / "layer16_monitor_metrics.csv", index=False)

    pd.DataFrame(
        {
            "method": ["sae_lr", "sae_lr", "random", "random"],
            "fpr": [0.0, 1.0, 0.0, 1.0],
            "tpr": [0.0, 1.0, 0.0, 1.0],
            "threshold": [1.0, 0.0, 1.0, 0.0],
            "auroc": [0.8, 0.8, 0.5, 0.5],
        }
    ).to_csv(results / "monitor" / "layer16_roc_points.csv", index=False)

    cat_rows = []
    for c in ["collision", "excessive_force", "boundary_violation", "high_approach_speed", "object_drop"]:
        cat_rows.extend(
            [
                {"category": c, "fpr": 0.0, "tpr": 0.0, "threshold": 1.0, "auroc": 0.7},
                {"category": c, "fpr": 1.0, "tpr": 1.0, "threshold": 0.0, "auroc": 0.7},
            ]
        )
    pd.DataFrame(cat_rows).to_csv(results / "monitor" / "layer16_per_category_roc.csv", index=False)

    pd.DataFrame(
        {
            "method": ["sae_lr"] * 5,
            "category": ["collision", "excessive_force", "boundary_violation", "high_approach_speed", "object_drop"],
            "auroc": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    ).to_csv(results / "monitor" / "layer16_per_category_auroc.csv", index=False)

    pd.DataFrame(
        {
            "threshold": [0.0, 0.5, 1.0],
            "success_rate": [0.4, 0.6, 0.8],
            "safety_violation_rate": [0.1, 0.2, 0.4],
        }
    ).to_csv(results / "monitor" / "layer16_pareto.csv", index=False)

    pd.DataFrame(
        {
            "num_features": [1, 5, 10],
            "collision_rate": [0.4, 0.3, 0.2],
            "scale": [0.0, 0.0, 0.0],
        }
    ).to_csv(results / "causal" / "layer16_causal_validation.csv", index=False)

    pd.DataFrame({"dict_size": [16000, 32000, 64000], "auroc": [0.7, 0.8, 0.82]}).to_csv(
        results / "ablations" / "dictionary_size.csv", index=False
    )
    pd.DataFrame({"layer": [16, 24], "collision_rate": [0.25, 0.22]}).to_csv(
        results / "ablations" / "layer_comparison.csv", index=False
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.run(
        [
            "python",
            "scripts/generate_figures.py",
            "--results_dir",
            str(results),
            "--output_dir",
            str(figures),
            "--paper_dir",
            str(paper),
            "--layer",
            "16",
        ],
        check=True,
        env=env,
    )

    assert (figures / "figure1_architecture.pdf").exists()
    assert (figures / "figure3_volcano.pdf").exists()
    assert (figures / "figure4_roc.pdf").exists()
    assert (figures / "figure5_clamp_pareto.pdf").exists()
    assert (figures / "figure6_ablations.pdf").exists()
    assert (paper / "tables" / "table_main_results.tex").exists()
    assert (paper / "tables" / "table_category_auroc.tex").exists()
    assert (paper / "tables" / "table_ablations.tex").exists()

    cat_table = (paper / "tables" / "table_category_auroc.tex").read_text(encoding="utf-8")
    assert "collision & 1 & 0.111 & 0.900" in cat_table
    assert "excessive\\_force & 2 & -0.222 & 0.800" in cat_table
