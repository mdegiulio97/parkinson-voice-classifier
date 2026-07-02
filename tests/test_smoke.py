"""Smoke test OFFLINE: la pipeline gira su dati sintetici, senza rete.

La CI non ha accesso al dataset reale UCI, quindi tutti i test usano
`make_synthetic`, che ha la stessa forma/colonne del reale ma con
ground-truth separabile.
"""

import numpy as np

from parkinson_clf.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic
from parkinson_clf.evaluate import classification_metrics, evaluate_model
from parkinson_clf.model import build_models, split_data


def test_synthetic_shape_and_columns():
    df = make_synthetic(n=195, seed=0)
    assert df.shape == (195, len(FEATURE_COLUMNS) + 1)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


def test_synthetic_is_reproducible():
    a = make_synthetic(n=100, seed=42)
    b = make_synthetic(n=100, seed=42)
    assert a.equals(b)


def test_split_is_stratified():
    df = make_synthetic(n=200, seed=1)
    x_train, x_test, y_train, y_test = split_data(df, test_size=0.3, seed=1)
    assert len(x_train) + len(x_test) == 200
    # proporzione della classe positiva simile tra train e test (stratificato)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_metrics_on_separable_scores():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, prob)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert set(m) == {"accuracy", "f1", "roc_auc"}


def test_build_models_has_three_plus_baseline():
    models = build_models(seed=0)
    assert "baseline_most_frequent" in models
    classifiers = [k for k in models if k != "baseline_most_frequent"]
    assert len(classifiers) == 3


def test_models_beat_baseline_on_separable_data():
    df = make_synthetic(n=300, seed=7)
    x_train, x_test, y_train, y_test = split_data(df, test_size=0.3, seed=7)
    models = build_models(seed=7)

    base = evaluate_model(models["baseline_most_frequent"], x_train, y_train, x_test, y_test)
    for name in ("logistic_regression", "random_forest", "svc_rbf"):
        m = evaluate_model(models[name], x_train, y_train, x_test, y_test)
        # su dati separabili ogni classificatore deve avere ROC-AUC alta e
        # battere la baseline (che per costruzione ha roc_auc ~= 0.5)
        assert m["roc_auc"] > 0.9
        assert m["roc_auc"] > base["roc_auc"]
