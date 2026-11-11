"""Train and evaluate the models; write results/metrics.json + figures.

Run (after downloading):
    uv run python scripts/download_data.py
    uv run python scripts/train.py

Main result: repeated subject-grouped cross-validation (no subject in both train
and test). For comparison the old record-level 70/30 split is also reported: it
puts recordings of the same subject on both sides and inflates the scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interattivo per ambienti headless / CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from parkinson_clf.data import FEATURE_COLUMNS, TARGET_COLUMN, load_parkinsons_with_groups
from parkinson_clf.evaluate import classification_metrics, grouped_cv_metrics
from parkinson_clf.model import build_models, split_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

SEED = 0
N_SPLITS = 5
N_REPEATS = 10
# Readable names for the figures.
PRETTY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svc_rbf": "SVC (RBF)",
    "baseline_most_frequent": "Baseline (most frequent)",
}


def _barplot(subject: dict[str, dict], leaky: dict[str, dict], path: Path) -> None:
    """Bar plot: subject-level ROC-AUC (mean +/- std) vs leaky record-level ROC-AUC."""
    names = list(subject)
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        [subject[n]["roc_auc"]["mean"] for n in names],
        width,
        yerr=[subject[n]["roc_auc"]["std"] for n in names],
        capsize=4,
        label="Subject-level CV (correct)",
    )
    ax.bar(
        x + width / 2,
        [leaky[n]["roc_auc"] for n in names],
        width,
        alpha=0.5,
        label="Record-level split (leaky)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("UCI Parkinsons - subject-level vs record-level evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _roc_curves(models: dict, x_test: np.ndarray, y_test: np.ndarray, path: Path) -> None:
    """ROC curves on the leaky record-level split (baseline excluded)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        if name == "baseline_most_frequent":
            continue
        prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = classification_metrics(y_test, prob)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{PRETTY[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - record-level split (leaky, for comparison)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df, groups = load_parkinsons_with_groups(DATA_DIR)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    n_subjects = len(set(groups))
    print(f"Dataset: {len(df)} recordings from {n_subjects} subjects, {x.shape[1]} features.")
    print(f"Class balance (status): {df['status'].value_counts().to_dict()}")

    models = build_models(seed=SEED)
    subject: dict[str, dict] = {}
    for name in models:
        subject[name] = grouped_cv_metrics(
            name, x, y, groups, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED
        )
        m = subject[name]
        print(
            f"[subject-level] {name:24s} roc_auc={m['roc_auc']['mean']:.3f}"
            f"+/-{m['roc_auc']['std']:.3f}  acc={m['accuracy']['mean']:.3f}"
            f"  f1={m['f1']['mean']:.3f}"
        )

    x_train, x_test, y_train, y_test = split_data(df, test_size=0.3, seed=SEED)
    leaky: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        leaky[name] = classification_metrics(y_test, model.predict_proba(x_test)[:, 1])
        print(f"[record-level, leaky] {name:24s} roc_auc={leaky[name]['roc_auc']:.3f}")

    out = {
        "dataset": "UCI Parkinsons (Little et al., 2009)",
        "n_samples": int(len(df)),
        "n_subjects": int(n_subjects),
        "n_features": int(x.shape[1]),
        "class_balance_positive_frac": float(y.mean()),
        "subject_level_cv": {
            "method": "repeated StratifiedGroupKFold by subject, pooled out-of-fold predictions",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "seed": SEED,
            "models": subject,
        },
        "record_level_split_leaky": {
            "method": "stratified 70/30 split by recording (same subject in train and test)",
            "test_size": 0.3,
            "seed": SEED,
            "models": leaky,
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Written: {RESULTS / 'metrics.json'}")

    _barplot(subject, leaky, RESULTS / "metrics_comparison.png")
    _roc_curves(models, x_test, y_test, RESULTS / "roc_curves.png")
    print(f"Figures saved in: {RESULTS}")


if __name__ == "__main__":
    main()
