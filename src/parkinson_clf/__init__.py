"""Classificazione di Parkinson da misure vocali (UCI Parkinsons).

Package del mini-progetto: caricamento dati (reale + sintetico), modelli
scikit-learn con baseline, metriche di classificazione binaria.
"""

from parkinson_clf.data import FEATURE_COLUMNS, load_parkinsons, make_synthetic
from parkinson_clf.evaluate import classification_metrics
from parkinson_clf.model import build_models, split_data

__all__ = [
    "FEATURE_COLUMNS",
    "load_parkinsons",
    "make_synthetic",
    "classification_metrics",
    "build_models",
    "split_data",
]
