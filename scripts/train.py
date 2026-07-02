"""Addestra e valuta i modelli, scrive results/metrics.json + figure.

Esegui (dopo il download):
    uv run python scripts/download_data.py
    uv run python scripts/train.py

Pipeline: carica UCI Parkinsons -> split stratificato -> addestra i 3
classificatori + baseline -> metriche sul test -> salva JSON e figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non interattivo per ambienti headless / CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from parkinson_clf.data import load_parkinsons
from parkinson_clf.evaluate import classification_metrics
from parkinson_clf.model import build_models, split_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

SEED = 0
# Nomi leggibili per le figure.
PRETTY = {
    "logistic_regression": "Regressione logistica",
    "random_forest": "Random forest",
    "svc_rbf": "SVC (RBF)",
    "baseline_most_frequent": "Baseline (most frequent)",
}


def _barplot(metrics: dict[str, dict], path: Path) -> None:
    """Barplot di confronto: accuracy / F1 / ROC-AUC per ciascun modello."""
    names = list(metrics)
    keys = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(keys):
        vals = [metrics[n][key] for n in names]
        ax.bar(x + (i - 1) * width, vals, width, label=key.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valore metrica")
    ax.set_title("Confronto modelli — UCI Parkinsons (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _roc_curves(models: dict, x_test: np.ndarray, y_test: np.ndarray, path: Path) -> None:
    """Curve ROC dei modelli con predict_proba non degenere (baseline esclusa)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        if name == "baseline_most_frequent":
            continue
        prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = classification_metrics(y_test, prob)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{PRETTY[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Caso (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curve ROC — UCI Parkinsons (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = load_parkinsons(DATA_DIR)
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1] - 1} feature.")
    print(f"Distribuzione classi (status): {df['status'].value_counts().to_dict()}")

    x_train, x_test, y_train, y_test = split_data(df, test_size=0.3, seed=SEED)
    models = build_models(seed=SEED)

    metrics: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        metrics[name] = classification_metrics(y_test, prob)
        m = metrics[name]
        print(
            f"{name:24s}  acc={m['accuracy']:.3f}  "
            f"f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}"
        )

    out = {
        "dataset": "UCI Parkinsons (Little et al., 2009)",
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "test_size": 0.3,
        "seed": SEED,
        "class_balance_positive_frac": float(df["status"].mean()),
        "models": metrics,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Scritto: {RESULTS / 'metrics.json'}")

    _barplot(metrics, RESULTS / "metrics_comparison.png")
    _roc_curves(models, x_test, y_test, RESULTS / "roc_curves.png")
    print(f"Figure salvate in: {RESULTS}")


if __name__ == "__main__":
    main()
