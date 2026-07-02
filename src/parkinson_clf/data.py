"""Caricamento dati per il classificatore Parkinson.

Due sorgenti, stessa forma (colonne feature + `status`):
- `load_parkinsons(data_dir)`  legge il CSV UCI scaricato in `data/`.
- `make_synthetic(n, seed)`    genera un dataset sintetico separabile con la
  stessa forma, usato dai test per girare OFFLINE (senza rete).

Il dataset UCI Parkinsons ha 22 feature di misure vocali (dysphonia) piu' la
colonna target `status` (1 = Parkinson, 0 = sano). La colonna `name` viene
scartata perche' e' un identificativo, non una feature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Le 22 feature di misure vocali del dataset UCI Parkinsons (esclusi `name` e `status`).
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


def load_parkinsons(data_dir: str | Path) -> pd.DataFrame:
    """Legge il CSV UCI (`parkinsons.data`) da `data_dir` e scarta `name`.

    Restituisce un DataFrame con le 22 feature + la colonna `status`.
    """
    csv_path = Path(data_dir) / "parkinsons.data"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {csv_path}. "
            "Esegui prima 'uv run python scripts/download_data.py'."
        )
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["name"])
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def make_synthetic(n: int = 195, seed: int = 0) -> pd.DataFrame:
    """Genera un dataset sintetico con la stessa forma del reale.

    Due cluster gaussiani separabili (classe 1 = Parkinson ~75%, come nel dataset
    reale sbilanciato). Serve ai test per validare la pipeline SENZA rete.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(0.75 * n))
    n_neg = n - n_pos
    n_features = len(FEATURE_COLUMNS)

    # Centri distinti per le due classi -> problema linearmente separabile.
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
