"""Metriche di classificazione binaria e valutazione dei modelli.

Con classi sbilanciate (~75% Parkinson) l'accuracy da sola inganna: riportiamo
sempre anche F1 e ROC-AUC. La ROC-AUC misura la capacita' di ranking ed e'
insensibile alla soglia, per questo e' la metrica piu' informativa qui.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def classification_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict:
    """Accuracy, F1 e ROC-AUC data la probabilita' della classe positiva (Parkinson)."""
    y_true = np.asarray(y_true).ravel()
    prob_pos = np.asarray(prob_pos).ravel()
    pred = (prob_pos >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)),
    }


def evaluate_model(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Addestra `model` sul train e restituisce le metriche sul test."""
    model.fit(x_train, y_train)
    prob_pos = model.predict_proba(x_test)[:, 1]
    return classification_metrics(y_test, prob_pos)
