"""Binary classifiers + baseline.

Three scikit-learn classifiers (logistic regression, random forest, RBF SVC)
plus a `DummyClassifier(most_frequent)` baseline: any useful model must beat it,
especially with imbalanced classes (~75% Parkinson's), where always predicting
the majority class already gives high accuracy.

Each model is a `Pipeline` with `StandardScaler`, so that voice features on very
different scales (Hz, jitter, shimmer, ...) are standardised inside each fit.
Scaling is irrelevant for the random forest but harmless, and keeps the
interface uniform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from parkinson_clf.data import FEATURE_COLUMNS, TARGET_COLUMN


def split_data(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split BY RECORDING. Leaky: kept only for comparison."""
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    """Build the three classifiers + the baseline, all with StandardScaler.

    Returns a dict name -> unfitted Pipeline. Every model exposes
    `predict_proba`, needed for ROC-AUC.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=seed)),
            ]
        ),
        "svc_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
            ]
        ),
        "baseline_most_frequent": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
    }
