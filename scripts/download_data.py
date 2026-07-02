"""Scarica il dataset UCI Parkinsons in data/ — idempotente.

Fonte: UCI Machine Learning Repository, dataset "Parkinsons" (Little et al.).
Licenza: CC BY 4.0. Riferimento: Little MA et al., "Suitability of dysphonia
measurements for telemonitoring of Parkinson's disease", IEEE TBME 2009,
DOI 10.1109/TBME.2008.2005954.

Regole del portfolio:
- I dati NON vengono mai committati (vedi .gitignore).
- Idempotente: se il file esiste gia', non riscarica.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / "parkinsons.data"
    if dest.exists():
        print(f"Gia' presente: {dest} ({dest.stat().st_size} byte). Nessun download.")
        return
    print(f"Scarico da: {URL}")
    urllib.request.urlretrieve(URL, dest)  # noqa: S310 (URL fisso e fidato UCI)
    print(f"Salvato: {dest} ({dest.stat().st_size} byte).")


if __name__ == "__main__":
    main()
