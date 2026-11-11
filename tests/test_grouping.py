"""Subject-level evaluation: parsing subject IDs and grouped cross-validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import parkinson_clf.evaluate as ev
from parkinson_clf.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic, subject_ids
from parkinson_clf.evaluate import grouped_cv_metrics


def _synthetic_with_subjects(n_subjects=30, per_subject=6, seed=0):
    df = make_synthetic(n=n_subjects * per_subject, seed=seed)
    df = df.sort_values(TARGET_COLUMN, kind="stable").reset_index(drop=True)
    groups = np.repeat([f"S{i:02d}" for i in range(n_subjects)], per_subject)
    # one class per subject: give each subject the majority label of its rows
    labels = df.groupby(groups)[TARGET_COLUMN].agg(lambda s: int(s.mean() >= 0.5))
    df[TARGET_COLUMN] = [labels[g] for g in groups]
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=int)
    return x, y, groups


def test_subject_ids_parses_uci_names():
    names = pd.Series(["phon_R01_S01_1", "phon_R01_S01_2", "phon_R01_S50_6"])
    assert subject_ids(names).tolist() == ["S01", "S01", "S50"]


def test_subject_ids_rejects_bad_name():
    with pytest.raises(ValueError):
        subject_ids(pd.Series(["phon_R01_S01_1", "foo"]))


def test_grouped_cv_never_shares_subject(monkeypatch):
    seen = []

    class RecordingSGKF(StratifiedGroupKFold):
        def split(self, X, y=None, groups=None):
            for tr, te in super().split(X, y, groups):
                seen.append((set(np.asarray(groups)[tr]), set(np.asarray(groups)[te])))
                yield tr, te

    monkeypatch.setattr(ev, "StratifiedGroupKFold", RecordingSGKF)
    x, y, groups = _synthetic_with_subjects()
    grouped_cv_metrics("logistic_regression", x, y, groups, n_splits=5, n_repeats=2)
    assert len(seen) == 10
    for tr_groups, te_groups in seen:
        assert tr_groups.isdisjoint(te_groups)


def test_grouped_cv_returns_mean_std():
    x, y, groups = _synthetic_with_subjects()
    m = grouped_cv_metrics("svc_rbf", x, y, groups, n_splits=5, n_repeats=3)
    assert set(m) == {"accuracy", "f1", "roc_auc"}
    for v in m.values():
        assert set(v) == {"mean", "std"}
    assert m["roc_auc"]["mean"] > 0.9
