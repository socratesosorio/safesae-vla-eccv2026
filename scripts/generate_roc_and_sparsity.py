"""
Generate ROC curve figure and sparsity curve from local progress analysis data.
Reads episode feature means and labels, trains monitors, and plots.
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
FEATURE_CSV = REPO / "logs" / "safesae_progress_sae_analysis" / "episode_feature_means_sae16384.csv"
LABEL_CSV = REPO / "logs" / "safesae_progress_labels" / "progress_labels_full.csv"
TOP20_CSV = REPO / "logs" / "safesae_progress_sae_analysis" / "top20_progress_features_sae16384.csv"
OUT_DIR = REPO / "paper" / "figures"


def load_top20_features():
    features = []
    with open(TOP20_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(int(row["feature_idx"]))
    return features[:20]


def load_data():
    label_map = {}
    with open(LABEL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["episode_id"]
            label = int(row["label"])
            if label in (0, 1):
                label_map[eid] = label

    episode_features = {}
    with open(FEATURE_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            eid = row[0]
            if eid in label_map:
                feats = np.array([float(x) for x in row[2:]], dtype=np.float32)
                episode_features[eid] = feats

    common = sorted(set(episode_features.keys()) & set(label_map.keys()))
    X = np.stack([episode_features[e] for e in common])
    y = np.array([label_map[e] for e in common])
    return X, y


def generate_roc_figure(X, y, top20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    configs = [
        ("SAE Feature LR (16384-d)", list(range(X.shape[1]))),
        ("Top-20 Feature LR", top20),
    ]

    colors = ["#2166ac", "#b2182b", "#4dac26", "#e66101"]
    for idx, (name, indices) in enumerate(configs):
        all_probs = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X[np.ix_(train_idx, indices)], y[train_idx])
            all_probs[test_idx] = clf.predict_proba(X[np.ix_(test_idx, indices)])[:, 1]

        fpr, tpr, _ = roc_curve(y, all_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx], lw=2.0,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance (0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Progress Classification ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "figure4_roc_curves"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}.pdf/png")


def generate_sparsity_curve(X, y, top20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy.stats import mannwhitneyu

    # Compute composite scores for all features
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        low = X[y == 0, i]
        high = X[y == 1, i]
        if np.std(low) == 0 and np.std(high) == 0:
            continue
        try:
            stat, pval = mannwhitneyu(low, high, alternative="two-sided")
            n1, n2 = len(low), len(high)
            r = 1 - 2 * stat / (n1 * n2)
            scores[i] = abs(r) * (-np.log10(max(pval, 1e-300)))
        except Exception:
            continue

    ranked = np.argsort(scores)[::-1]
    k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 1881]
    aucs = []

    for k in k_values:
        if k > X.shape[1]:
            k = X.shape[1]
        indices = ranked[:k]
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for train_idx, test_idx in kf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X[np.ix_(train_idx, indices)], y[train_idx])
            probs = clf.predict_proba(X[np.ix_(test_idx, indices)])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], probs))
        aucs.append(np.mean(fold_aucs))

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    ax.plot(k_values, aucs, "o-", color="#2166ac", lw=2, markersize=6)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Number of top-ranked features")
    ax.set_ylabel("AUROC")
    ax.set_title("Sparsity-Performance Curve")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.45, 1.0])

    fig.tight_layout()
    out_path = OUT_DIR / "figure10_sparsity_curve_computed"
    fig.savefig(str(out_path) + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}.pdf/png")

    # Save data
    data = {"k_values": k_values, "aucs": [round(a, 4) for a in aucs]}
    with open(OUT_DIR / "sparsity_curve_data.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    print("Loading data...")
    X, y = load_data()
    top20 = load_top20_features()
    print(f"  {len(X)} episodes, {X.shape[1]} features")

    print("Generating ROC curves...")
    generate_roc_figure(X, y, top20)

    print("Generating sparsity curve...")
    generate_sparsity_curve(X, y, top20)


if __name__ == "__main__":
    main()
