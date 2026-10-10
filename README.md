# Transaction Fraud Detection (PaySim)

Fraud detection on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
synthetic mobile-money dataset (~6.3M transactions, ~0.13% fraud).

## What this project demonstrates
- **Imbalanced classification**: models judged by PR-AUC, precision and recall, not accuracy.
- **Leakage-safe evaluation**: a time-based train/validation/test split on `step`.
- **Cost-based decision thresholds**: the operating point is chosen by the business
  cost of missed fraud versus false alarms.
- **Explainability**: SHAP explanations for global and per-transaction behaviour.

## Project structure
```
data/              raw data (not committed)
notebooks/         analysis notebooks (call code in src/)
src/               importable project code
reports/figures/   saved figures
tests/             pytest tests
```

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the CSV in `data/`.

## Status
Work in progress.
