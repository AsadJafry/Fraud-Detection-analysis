# CLAUDE.md: Project rules

Transaction fraud detection on the PaySim dataset (~6.3M rows, ~0.13% fraud).
This is a portfolio project aimed at bank hiring teams. It should demonstrate
imbalanced classification, leakage-safe evaluation, cost-based decision
thresholds and explainability.

## Data and evaluation
- **Time-based split only**, using the `step` column (1 step = 1 hour). Train on
  earlier steps and validate/test on later steps. Never use a random split or
  shuffle across time.
- **Never report accuracy as a headline metric.** With ~0.13% fraud, a model
  that always predicts "not fraud" is 99.87% accurate. Report PR-AUC (average
  precision), precision and recall, plus cost-based results at the chosen threshold.
- **Drop `isFlaggedFraud`.** It is the legacy rule's output, not a usable feature.
- **Never use `nameOrig` / `nameDest` as raw features.** They are high-cardinality
  IDs. Derived features (e.g. merchant vs customer prefix) are fine only if they
  are computed without using future information.
- Any statistic used for a feature (means, counts, encodings) must be fitted on
  the training window only.

## Code
- Fix random seeds to **42** everywhere (numpy, sklearn, xgboost, splits).
- Keep logic in `src/` as importable functions. Notebooks only call those
  functions and visualize results.
- Write readable code with docstrings. No overengineering: no extra abstraction
  layers, config frameworks or class hierarchies unless they are clearly needed.
- Add tests in `tests/` for core logic (splitting, feature building, cost/threshold math).

## Figures
- Save all figures to `reports/figures/` at **150 dpi**.
- Every figure needs a clear title, axis labels and a legend where relevant.

## Layout
- `data/`: raw and processed data (gitignored)
- `notebooks/`: exploration and presentation, calling `src/`
- `src/`: importable project code
- `reports/figures/`: saved figures
- `tests/`: pytest tests
