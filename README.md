# Parkinson's disease from voice measures — subject-level evaluation

> **University coursework — archived.** Kept as a worked example of subject-level evaluation; not maintained.

Binary classification **Parkinson's vs healthy** from sustained-vowel voice measures (UCI Parkinsons). Three scikit-learn classifiers against a majority-class baseline, evaluated so that **no subject appears in both training and test data**. Reproducible with uv, offline tests, CI.

![subject-level vs record-level](results/metrics_comparison.png)

## Results (subject-level cross-validation)

Repeated stratified group K-fold by subject (5 folds × 10 repeats, seed 0); metrics on pooled out-of-fold predictions, mean ± std across repeats. 195 recordings, 32 subject IDs.

| Model | Accuracy | F1 | ROC-AUC |
|--|:--:|:--:|:--:|
| Logistic regression | 0.782 ± 0.019 | 0.859 ± 0.014 | 0.727 ± 0.031 |
| Random forest | 0.774 ± 0.014 | 0.859 ± 0.010 | 0.777 ± 0.025 |
| SVC (RBF) | 0.777 ± 0.024 | 0.856 ± 0.018 | 0.643 ± 0.033 |
| Baseline (most frequent) | 0.754 ± 0.000 | 0.860 ± 0.000 | 0.500 ± 0.000 |

With ~75% positive recordings, accuracy and F1 are close to the baseline for every model; ROC-AUC is the informative metric. The best model (random forest) reaches about 0.78, with a small dataset and wide uncertainty.

## Why record-level splitting leaks

Each subject contributes about six recordings. A plain 70/30 split by recording puts recordings of the same voice on both sides, so the models partly recognise the speaker rather than the disease:

| Model | ROC-AUC, subject-level (correct) | ROC-AUC, record-level split (leaky) |
|--|:--:|:--:|
| Logistic regression | 0.727 | 0.882 |
| Random forest | 0.777 | 0.895 |
| SVC (RBF) | 0.643 | 0.908 |
| Baseline (most frequent) | 0.500 | 0.500 |

The leaky split inflates every model and even reverses the ranking: SVC looks best by recording and is worst by subject. An earlier version of this repository reported the leaky numbers.

## Pipeline

```
data/parkinsons.data ──► load_parkinsons_with_groups()  (subject ID parsed from `name`, e.g. phon_R01_S01_1 → S01)
                              │
             StratifiedGroupKFold by subject, repeated
                              │
   logistic regression · random forest · SVC (RBF) · baseline (most frequent)
                              ▼
          results/metrics.json  +  figures (subject vs record level, ROC)
```

Tests run offline on a synthetic generator with the same shape, and check that no subject is ever in both train and test folds.

## Dataset

**UCI Parkinsons** (Little et al., 2009): 195 sustained-vowel recordings, 22 voice features (jitter, shimmer, HNR/NHR, RPDE, DFA, PPE, …) and the target `status` (1 = Parkinson's, 0 = healthy). The UCI description reports 31 people; the `name` column contains 32 distinct subject IDs, which are used as groups.

- Source: UCI Machine Learning Repository (`parkinsons.data`), licence CC BY 4.0.
- Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* IEEE Trans Biomed Eng, 2009. DOI: [10.1109/TBME.2008.2005954](https://doi.org/10.1109/TBME.2008.2005954)

## How to run (uv)

```bash
uv sync --extra dev
uv run pytest -q                         # offline tests
uv run ruff check .
uv run python scripts/download_data.py   # UCI Parkinsons into data/
uv run python scripts/train.py           # evaluation, writes results/
uv run streamlit run app/main.py         # results dashboard
```

## Limitations

- Small dataset (32 subjects, only a handful healthy): estimates have high variance.
- No hyperparameter or threshold tuning.
- Not for clinical use.

## License

Code: MIT (see `LICENSE`). Data: UCI Parkinsons, CC BY 4.0.
