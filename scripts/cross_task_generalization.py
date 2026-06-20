"""
Cross-task generalization experiment.
Train on goal+object, test on spatial+long (and reverse).
Also runs random-split baseline for comparison.
Uses existing episode feature means from logs/.
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parent.parent
FEATURE_CSV = REPO / "logs" / "safesae_progress_sae_analysis" / "episode_feature_means_sae16384.csv"
LABEL_CSV = REPO / "logs" / "safesae_progress_labels" / "progress_labels_full.csv"
TOP20_CSV = REPO / "logs" / "safesae_progress_sae_analysis" / "top20_progress_features_sae16384.csv"
OUT_DIR = REPO / "logs" / "cross_task_generalization"


def load_top20_features():
    features = []
    with open(TOP20_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(int(row["feature_idx"]))
    return features[:20]


def load_data():
    suite_map = {}
    label_map = {}
    with open(LABEL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["episode_id"]
            label = int(row["label"])
            suite = row["suite"]
            if label in (0, 1):
                suite_map[eid] = suite
                label_map[eid] = label

    episode_features = {}
    feature_names = None
    with open(FEATURE_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[2:]
        for row in reader:
            eid = row[0]
            if eid in label_map:
                feats = np.array([float(x) for x in row[2:]], dtype=np.float32)
                episode_features[eid] = feats

    common = set(episode_features.keys()) & set(label_map.keys())
    episodes = sorted(common)
    X = np.stack([episode_features[e] for e in episodes])
    y = np.array([label_map[e] for e in episodes])
    suites = np.array([suite_map[e] for e in episodes])
    return X, y, suites, feature_names


def eval_split(X_train, y_train, X_test, y_test, indices):
    X_tr = X_train[:, indices]
    X_te = X_test[:, indices]
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_tr, y_train)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
    return {
        "auroc": round(auc, 4),
        "f1": round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, suites, feature_names = load_data()
    print(f"  {len(X)} episodes, {X.shape[1]} features")
    for s in sorted(set(suites)):
        n = (suites == s).sum()
        pos = y[suites == s].sum()
        print(f"  {s}: {n} episodes ({int(pos)} pos, {int(n - pos)} neg)")

    top20 = load_top20_features()
    all_indices = list(range(X.shape[1]))
    configs = [
        ("full_lr", all_indices),
        ("top20_lr", top20),
    ]

    all_results = {}

    # --- Cross-task: train goal+object -> test spatial+long ---
    print("\n--- Cross-task: goal+object -> spatial+long ---")
    train_mask = np.isin(suites, ["goal", "object"])
    test_mask = np.isin(suites, ["spatial", "long"])
    for name, indices in configs:
        r = eval_split(X[train_mask], y[train_mask], X[test_mask], y[test_mask], indices)
        r["train_suites"] = "goal+object"
        r["test_suites"] = "spatial+long"
        all_results[f"go_to_sl_{name}"] = r
        print(f"  {name}: AUROC={r['auroc']}, F1={r['f1']}")

    # --- Cross-task: train spatial+long -> test goal+object ---
    print("\n--- Cross-task: spatial+long -> goal+object ---")
    train_mask2 = np.isin(suites, ["spatial", "long"])
    test_mask2 = np.isin(suites, ["goal", "object"])
    for name, indices in configs:
        r = eval_split(X[train_mask2], y[train_mask2], X[test_mask2], y[test_mask2], indices)
        r["train_suites"] = "spatial+long"
        r["test_suites"] = "goal+object"
        all_results[f"sl_to_go_{name}"] = r
        print(f"  {name}: AUROC={r['auroc']}, F1={r['f1']}")

    # --- Random-split baseline (3 seeds) ---
    print("\n--- Random-split baseline (mean of 3 seeds) ---")
    for name, indices in configs:
        aucs = []
        for seed in [42, 123, 456]:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            for tr_idx, te_idx in sss.split(X, y):
                r = eval_split(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], indices)
                aucs.append(r["auroc"])
        mean_auc = round(np.mean(aucs), 4)
        std_auc = round(np.std(aucs), 4)
        all_results[f"random_{name}"] = {
            "auroc_mean": mean_auc,
            "auroc_std": std_auc,
            "aurocs": aucs,
        }
        print(f"  {name}: AUROC={mean_auc} +/- {std_auc}")

    # Save JSON
    out_json = OUT_DIR / "cross_task_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_json}")

    # Generate latex table
    latex = (
        "\\begin{tabular}{llcccc}\\toprule\n"
        "Split & Method & AUROC$\\uparrow$ & F1$\\uparrow$ & "
        "Train N & Test N \\\\ \\midrule\n"
    )
    rows = [
        ("goal+obj $\\to$ spat+long", "go_to_sl"),
        ("spat+long $\\to$ goal+obj", "sl_to_go"),
        ("Random (mean$\\pm$std)", "random"),
    ]
    for split_label, prefix in rows:
        for method, method_label in [("full_lr", "SAE LR (16K)"), ("top20_lr", "Top-20 LR")]:
            key = f"{prefix}_{method}"
            r = all_results[key]
            if "auroc_mean" in r:
                latex += (
                    f"{split_label} & {method_label} & "
                    f"{r['auroc_mean']:.3f}$\\pm${r['auroc_std']:.3f} & -- & -- & -- \\\\\n"
                )
            else:
                latex += (
                    f"{split_label} & {method_label} & "
                    f"{r['auroc']:.3f} & {r['f1']:.3f} & {r['train_n']} & {r['test_n']} \\\\\n"
                )
        if prefix != "random":
            latex += "\\midrule\n"

    latex += "\\bottomrule\\end{tabular}"
    out_tex = OUT_DIR / "table_cross_task.tex"
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"Saved table to {out_tex}")


if __name__ == "__main__":
    main()
