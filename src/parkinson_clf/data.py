"""Data loading for the Parkinson classifier.

Two sources with the same shape (feature columns + `status`):
- `load_parkinsons(data_dir)`              reads the UCI CSV downloaded into `data/`.
- `load_parkinsons_with_groups(data_dir)`  same, plus the subject ID of each row.
- `make_synthetic(n, seed)`                separable synthetic data with the same
  shape, used by the tests to run OFFLINE.

UCI Parkinsons has 195 sustained-vowel recordings from 31 subjects (about six
per subject), 22 voice features and the target `status` (1 = Parkinson's,
0 = healthy). The `name` column (e.g. `phon_R01_S01_1`) identifies the subject:
it is not a feature, but it is needed to keep each subject's recordings on the
same side of every split.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# The 22 voice features of UCI Parkinsons (excluding `name` and `status`).
FEATURE_COLUMNS = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]
TARGET_COLUMN = "status"
_NAME_RE = re.compile(r"^phon_R01_(S\d+)_\d+$")


def subject_ids(names: pd.Series) -> np.ndarray:
    """Subject ID from each UCI recording name (`phon_R01_S01_1` -> `S01`).

    Raises ValueError on any name that does not match, rather than inventing a
    subject.
    """
    out = []
    for name in names:
        m = _NAME_RE.match(str(name))
        if m is None:
            raise ValueError(f"Unexpected recording name: {name!r}")
        out.append(m.group(1))
    return np.array(out, dtype=object)


def _read_csv(data_dir: str | Path) -> pd.DataFrame:
    csv_path = Path(data_dir) / "parkinsons.data"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Run 'uv run python scripts/download_data.py' first."
        )
    return pd.read_csv(csv_path)


def load_parkinsons(data_dir: str | Path) -> pd.DataFrame:
    """Read the UCI CSV (`parkinsons.data`): the 22 features + `status`."""
    return _read_csv(data_dir)[FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_parkinsons_with_groups(data_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Like `load_parkinsons`, plus the subject ID of each row (for grouped CV)."""
    df = _read_csv(data_dir)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]], subject_ids(df["name"])


def make_synthetic(n: int = 195, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset with the same shape as the real one.

    Two separable Gaussian clusters (class 1 = Parkinson's, ~75%, as in the
    imbalanced real data). Lets the tests validate the pipeline WITHOUT network.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(0.75 * n))
    n_neg = n - n_pos
    n_features = len(FEATURE_COLUMNS)

    # Distinct centres for the two classes -> linearly separable problem.
    center_pos = rng.normal(0.0, 0.3, size=n_features) + 1.0
    center_neg = rng.normal(0.0, 0.3, size=n_features) - 1.0

    x_pos = rng.normal(center_pos, 1.0, size=(n_pos, n_features))
    x_neg = rng.normal(center_neg, 1.0, size=(n_neg, n_features))

    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    order = rng.permutation(n)
    df = pd.DataFrame(x[order], columns=FEATURE_COLUMNS)
    df[TARGET_COLUMN] = y[order]
    return df
