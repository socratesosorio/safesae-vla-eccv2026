# SafeSAE-VLA

Official code and result artifacts for:

**SafeSAE-VLA: Interpreting OpenVLA Progress Dynamics with Sparse Feature Analysis**
Socrates Osorio, Joy Zheyun Yang. ECCV 2026.

SafeSAE-VLA trains sparse autoencoders (SAEs) on the residual stream of a
vision-language-action policy (OpenVLA) and asks whether *relative task progress*
is recoverable from a small set of interpretable feature directions. On 750 LIBERO
episodes, progress is strongly and sparsely decodable (0.918 AUROC overall; 0.894
with 20 features), and the same directions support direct on-OpenVLA interventions.
Dense baselines are equal or stronger predictors. The contribution of the sparse
basis is inspectability and intervention compatibility, not peak accuracy.

---

## Repository layout

```
scripts/      Numbered, end-to-end pipeline (collection → SAE → analysis → figures)
src/          Library imported by the scripts (sae/, analysis/, monitor/, data/, utils/)
configs/      YAML configs for rollouts, SAE training, and evaluation
results/logs/ Result summaries (CSV/JSON/PDF) that back every table and figure
tests/        Unit / smoke tests
```

Large artifacts (OpenVLA checkpoints, trained SAE checkpoints, and cached rollout
activations) are **not** in this repo. See [Data & checkpoints](#data--checkpoints).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# requirements-local.txt pins a CPU-only analysis subset if you only want to
# re-run the statistics/figures from the cached result files.
```

## Data & checkpoints

The OpenVLA rollouts (residual-stream activations at layers 16/20/24) and the layer-20 SAE
checkpoint (`d_sae=16384, k=32`) are hosted on Hugging Face:

> **https://huggingface.co/datasets/socratesosorio/safesae-vla-eccv2026**

The per-suite OpenVLA policy weights
(`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,long}`) are already public on the
Hugging Face Hub.

Point the scripts at the downloaded data with `--data_dir` / `--sae_checkpoint`.
Every number and figure in the paper can be re-derived from the small summary files
already committed under `results/logs/` without re-running the GPU pipeline.

## Reproducing the paper

| Paper element | Script | Result artifact |
|---|---|---|
| Rollout collection + activation caching | `scripts/02_collect_rollouts.py` | — |
| SAE training (`d_sae=16384, k=32`) | `scripts/03_train_sae.py` | — |
| Progress relabeling (quartile split) | `scripts/09_compute_progress_labels.py` | `results/logs/safesae_progress_sae_analysis/` |
| Differential analysis (Mann–Whitney + BH-FDR), Table 4 / Fig 3 (volcano) / Fig 4 (heatmap) | `scripts/11_progress_sae_feature_analysis.py` | `.../volcano_*.pdf`, `.../heatmap_*.pdf` |
| Main monitor + sparsity, Tables 2–3, Fig 2 | `scripts/06_evaluate_monitor.py` | `results/logs/.../layer20_monitor_metrics.csv` |
| Per-suite AUROC, Table 5 | `scripts/11_progress_sae_feature_analysis.py` | `.../layer20_per_category_auroc.csv` |
| **Dense/raw baselines + split robustness, Table 8** | `scripts/29_generate_eccv_rebuttal_checks.py` | `results/logs/eccv_rebuttal_checks/rebuttal_progress_baselines_and_splits.csv` |
| **Layer sweep (16/20/24)** | `scripts/16_run_monitor_layer_sweep.py` | `results/logs/safesae_progress_raw_analysis/` |
| **Geometric-vs-semantic audit (0.471 vs 0.968)** | `scripts/47_run_success_labeled_baseline_audit.py`, `scripts/32_run_semantic_progress_audit.py` | `results/logs/eccv_success_labeled_baseline_audit_after676838/` |
| **Direct OpenVLA class-mean intervention (+0.763 vs +0.078)** | `scripts/33_run_openvla_class_mean_intervention.py`, `scripts/25_run_directional_offline_intervention.py` | `results/logs/workshop_strengthening/`, `results/logs/progress_feature_robustness/` |
| **Closed-loop specificity batch (1.00→0.50)** | `scripts/49_prepare_eccv_closed_loop_specificity_rescue.py`, `scripts/34_summarize_closed_loop_intervention.py` | `results/logs/eccv_closed_loop_specificity_rescue/` |
| Confound controls / same-task pairs / 32K SAE sanity | `scripts/39_generate_eccv_confound_controls.py`, `scripts/44_..._pairwise_signed_prefix_checks.py`, `scripts/46_..._32k_dictionary_sanity.py` | `results/logs/eccv_confound_controls_*/`, `results/logs/eccv_pairwise_signed_prefix_checks_*/`, `results/logs/eccv_32k_dictionary_sanity_*/` |
| Cross-model SAE-Scope (secondary support) | `scripts/05_causal_validation.py` | `results/logs/` |
| All paper figures | `scripts/07_generate_figures.py`, `scripts/27_generate_submission_figures.py` | — |

Scripts numbered above ~28 are the rebuttal/camera-ready additions. A handful of
other `scripts/*` are exploratory checks that did **not** make the paper (e.g.
alternative proxies with zero top-feature overlap, leave-one-suite-out ranking);
they are kept for transparency and are labeled as such in their docstrings.

## Quick validation (no GPU)

```bash
python scripts/01_test_pipeline.py --config configs/rollout.yaml --num_rollouts 50
```

## Citation

```bibtex
@inproceedings{osorio2026safesaevla,
  title     = {SafeSAE-VLA: Interpreting OpenVLA Progress Dynamics with Sparse Feature Analysis},
  author    = {Osorio, Socrates and Yang, Joy Zheyun},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). OpenVLA, LIBERO,
and Octo are the property of their respective authors and are subject to their own
licenses.
