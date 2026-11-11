"""Download the UCI Parkinsons dataset into data/ (idempotent).

Source: UCI Machine Learning Repository, "Parkinsons" (Little et al.), CC BY 4.0.
Little MA et al., "Suitability of dysphonia measurements for telemonitoring of
Parkinson's disease", IEEE TBME 2009, DOI 10.1109/TBME.2008.2005954.

Data are never committed (see .gitignore). If the file exists, nothing is done.
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
        print(f"Already present: {dest} ({dest.stat().st_size} bytes). Nothing to do.")
        return
    print(f"Downloading from: {URL}")
    urllib.request.urlretrieve(URL, dest)  # noqa: S310 (fixed, trusted UCI URL)
    print(f"Saved: {dest} ({dest.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
