# ✈️ Flight Delay Predictor

Predicts whether a US domestic flight will be delayed more than 15 minutes, using pre-flight information only (route, carrier, schedule, season). Built end-to-end: found and fixed a critical data bug in an inherited model, rebuilt the dataset from real government flight records, retrained and validated the models, and shipped an interactive prediction app.

## The problem

The existing model always predicted "delayed" — every input, every time. Root cause: its training data had been silently pre-filtered to contain only already-delayed flights (minimum recorded delay was 15 minutes; zero on-time flights existed anywhere in the dataset). The model wasn't broken — the data made "always delayed" the only pattern there was to learn. Both models were, provably, worse than a constant guess.

## What I did

- **Diagnosed the root cause** by inspecting the raw data distribution rather than trusting the reported metrics — accuracy numbers alone (41–53%) looked like an ML problem; the actual issue was a data problem hiding behind them.
- **Sourced and rebuilt the dataset** from real BTS (U.S. Bureau of Transportation Statistics) on-time performance records, verifying byte-identical schema across two different data sources before merging them.
- **Rebuilt the modeling pipeline** (EDA, feature engineering, leakage-safe feature selection, time-based train/test split) and retrained Logistic Regression and XGBoost classifiers on the corrected data.
- **Shipped a working product** — an interactive Streamlit app that takes a flight's details and returns a calibrated delay-risk prediction, with transparent accuracy reporting rather than a false sense of confidence.

## Results

| | Before | After |
|---|---|---|
| On-time flights in training data | 2.6% | 78.8% |
| XGBoost accuracy | 53.1% (worse than guessing) | **75.5%** |
| ROC-AUC | not meaningfully measurable | **0.618** |
| Real-world usability | Always predicts "delayed" | Predictions vary correctly with actual risk factors |

Modeled against a cost-sensitive business scenario (missed delays cost more than false alarms), the fixed model reduces cost by **19%** versus a no-prediction baseline.

## Tech stack

**Python** · **pandas / NumPy** · **scikit-learn** · **XGBoost** · **Streamlit** · **Jupyter**

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
final.ipynb / final.py   EDA + full modeling pipeline
build_dataset.py         Sources and merges the real BTS dataset
app.py                   Interactive prediction app (Streamlit)
models/                  Trained models and preprocessing artifacts
details.md               Full technical writeup
```

See [`details.md`](details.md) for the complete technical breakdown — data sources, feature engineering, model comparison, and every design decision behind this project.
