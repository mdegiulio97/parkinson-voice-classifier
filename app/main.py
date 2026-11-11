"""Parkinson's from voice measures - results dashboard (Streamlit).

Start with:  uv run streamlit run app/main.py
Shows the metrics and figures written by scripts/train.py. Generate them first:
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
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svc_rbf": "SVC (RBF)",
    "baseline_most_frequent": "Baseline (most frequent)",
}

st.set_page_config(page_title="Parkinson's - voice classification", layout="wide")
st.title("Parkinson's disease from voice measures")
st.caption(
    "Binary classification (Parkinson's vs healthy) on UCI Parkinsons, evaluated by "
    "subject: no subject appears in both training and test data."
)

if not METRICS.exists():
    st.info(
        "No results found. Generate them first:\n\n"
        "```\nuv run python scripts/download_data.py\n"
        "uv run python scripts/train.py\n```"
    )
    st.stop()

data = json.loads(METRICS.read_text(encoding="utf-8"))
subject = data["subject_level_cv"]
leaky = data["record_level_split_leaky"]

c1, c2, c3 = st.columns(3)
c1.metric("Recordings", data["n_samples"])
c2.metric("Subjects", data["n_subjects"])
c3.metric("Parkinson's fraction", f"{data['class_balance_positive_frac']:.2%}")

st.subheader(
    f"Subject-level cross-validation ({subject['n_splits']} folds x {subject['n_repeats']} repeats)"
)
rows = []
for name, m in subject["models"].items():
    rows.append(
        {
            "Model": PRETTY.get(name, name),
            "ROC-AUC (mean)": round(m["roc_auc"]["mean"], 3),
            "ROC-AUC (std)": round(m["roc_auc"]["std"], 3),
            "Accuracy": round(m["accuracy"]["mean"], 3),
            "F1": round(m["f1"]["mean"], 3),
            "ROC-AUC, record-level split (leaky)": round(leaky["models"][name]["roc_auc"], 3),
        }
    )
st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

col_a, col_b = st.columns(2)
bar = RESULTS / "metrics_comparison.png"
roc = RESULTS / "roc_curves.png"
if bar.exists():
    col_a.subheader("Subject-level vs record-level")
    col_a.image(str(bar), use_container_width=True)
if roc.exists():
    col_b.subheader("ROC curves (record-level split, leaky)")
    col_b.image(str(roc), use_container_width=True)

st.caption("Demonstration only, not for clinical use: small dataset, wide uncertainty.")
