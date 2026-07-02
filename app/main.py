"""Parkinson da misure vocali — cruscotto dei risultati (app Streamlit).

Avvia con:  uv run streamlit run app/main.py
Mostra metriche e figure prodotte da scripts/train.py. Genera prima i risultati:
    uv run python scripts/download_data.py
    uv run python scripts/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics.json"

PRETTY = {
    "logistic_regression": "Regressione logistica",
    "random_forest": "Random forest",
    "svc_rbf": "SVC (RBF)",
    "baseline_most_frequent": "Baseline (most frequent)",
}

st.set_page_config(page_title="Parkinson · Classificazione vocale", layout="wide")
st.title("🧠 Parkinson: classificazione da misure vocali")
st.caption(
    "Classificazione binaria (Parkinson vs sano) su UCI Parkinsons. "
    "Tre classificatori scikit-learn + baseline; con classi sbilanciate la "
    "metrica chiave e' la ROC-AUC."
)

if not METRICS.exists():
    st.info(
        "Nessun risultato trovato. Genera prima le metriche:\n\n"
        "```\nuv run python scripts/download_data.py\n"
        "uv run python scripts/train.py\n```"
    )
    st.stop()

data = json.loads(METRICS.read_text(encoding="utf-8"))

c1, c2, c3 = st.columns(3)
c1.metric("Campioni", data["n_samples"])
c2.metric("Feature vocali", data["n_features"])
c3.metric("Frazione Parkinson", f"{data['class_balance_positive_frac']:.2%}")

st.subheader("Metriche sul test set")
rows = []
for name, m in data["models"].items():
    rows.append(
        {
            "Modello": PRETTY.get(name, name),
            "Accuracy": round(m["accuracy"], 3),
            "F1": round(m["f1"], 3),
            "ROC-AUC": round(m["roc_auc"], 3),
        }
    )
df = pd.DataFrame(rows).set_index("Modello")
st.dataframe(df.style.highlight_max(axis=0, color="#2e7d32"), use_container_width=True)

col_a, col_b = st.columns(2)
bar = RESULTS / "metrics_comparison.png"
roc = RESULTS / "roc_curves.png"
if bar.exists():
    col_a.subheader("Confronto metriche")
    col_a.image(str(bar), use_container_width=True)
if roc.exists():
    col_b.subheader("Curve ROC")
    col_b.image(str(roc), use_container_width=True)

st.caption(
    "Uso dimostrativo/educativo. Nessun impiego clinico: il dataset e' piccolo "
    "(195 soggetti) e la valutazione e' un singolo split."
)
