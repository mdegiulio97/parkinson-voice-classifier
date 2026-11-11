"""Binary classification metrics and model evaluation.

With imbalanced classes (~75% Parkinson's) accuracy alone is misleading: F1 and
ROC-AUC are always reported too. ROC-AUC measures ranking and does not depend on
the threshold, so it is the most informative metric here.

The main evaluation is subject-level: every subject's recordings stay on the
same side of each split (`grouped_cv_metrics`). Splitting by recording instead
puts the same voice in train and test and inflates the scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from parkinson_clf.model import build_models


def classification_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict:
    """Accuracy, F1 and ROC-AUC given the positive-class (Parkinson's) probability."""
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
    """Fit `model` on train and return the test metrics."""
    model.fit(x_train, y_train)
    prob_pos = model.predict_proba(x_test)[:, 1]
    return classification_metrics(y_test, prob_pos)


def grouped_cv_metrics(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Repeated subject-grouped CV; metrics on pooled out-of-fold predictions.

    Each repeat reshuffles a StratifiedGroupKFold (no subject in both train and
    test), collects out-of-fold probabilities from a fresh copy of the model and
    computes the metrics on all of them at once. Returns mean and std across
    repeats for accuracy, F1 and ROC-AUC.
    """
    template = build_models(seed)[model_name]
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)
    per_repeat: dict[str, list[float]] = {"accuracy": [], "f1": [], "roc_auc": []}
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y), dtype=float)
        for tr, te in cv.split(x, y, groups):
            model = clone(template)
            model.fit(x[tr], y[tr])
            oof[te] = model.predict_proba(x[te])[:, 1]
        for k, v in classification_metrics(y, oof).items():
            per_repeat[k].append(v)
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per_repeat.items()
    }
