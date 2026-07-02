"""Modelli di classificazione binaria + baseline.

Tre classificatori scikit-learn (regressione logistica, random forest, SVC RBF)
piu' una baseline `DummyClassifier(most_frequent)` che serve da riferimento
onesto: qualunque modello utile deve batterla, specie con classi sbilanciate
(~75% Parkinson) dove predire sempre la maggioranza da' comunque alta accuracy.

Ogni modello e' una `Pipeline` con `StandardScaler`, cosi' le feature vocali
(scale molto diverse: Hz, jitter, shimmer, ...) sono standardizzate prima del
fit. Per random forest lo scaling e' irrilevante ma innocuo, e mantiene
l'interfaccia uniforme.
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
    """Split stratificato train/test (mantiene le proporzioni delle classi)."""
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def build_models(seed: int = 0) -> dict[str, Pipeline]:
    """Costruisce i tre classificatori + la baseline, tutti con StandardScaler.

    Restituisce un dizionario nome -> Pipeline non ancora addestrata. Tutti i
    modelli espongono `predict_proba` per calcolare la ROC-AUC.
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
