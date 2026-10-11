"""SHAP explanations for the XGBoost fraud model, globally and per transaction.

SHAP values are in log-odds units: for each transaction,
``base value + sum of SHAP values = model margin``, and the fraud score is
``sigmoid(margin)``. A positive value pushes towards "fraud".
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_features
from src.models import MAX_PAYSIM_AMOUNT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADLINE_MODEL = "xgb_pre_transaction"
HEADLINE_NAME = "XGBoost (pre-transaction)"

# Readable names for plots and explanations.
FEATURE_LABELS = {
    "amount_log": "Amount (log)",
    "oldbalanceOrg_log": "Origin opening balance (log)",
    "oldbalanceDest_log": "Destination opening balance (log)",
    "amount_to_balance_ratio": "Amount / origin balance",
    "dest_new_or_zero": "Destination balance was 0",
    "hour": "Hour of day",
    "is_transfer": "Is TRANSFER (vs CASH_OUT)",
}

# Columns recorded after the transaction; not needed by the pre-transaction model.
_POST_TRANSACTION_COLUMNS = ["newbalanceOrig", "newbalanceDest"]


def readable_values(X: pd.DataFrame) -> pd.DataFrame:
    """Undo the log transform on amount and balance features for display.

    Waterfall plots then show "Amount = 181" rather than "Amount (log) = 5.2".
    """
    out = X.copy()
    for col in ["amount_log", "oldbalanceOrg_log", "oldbalanceDest_log"]:
        if col in out:
            out[col] = np.round(np.expm1(out[col]), 2)
    if "amount_to_balance_ratio" in out:
        out["amount_to_balance_ratio"] = out["amount_to_balance_ratio"].round(4)
    return out


def best_iteration_booster(model):
    """Return the model's booster cut to its best early-stopped iteration.

    A model trained with early stopping keeps the extra trees grown after the
    best round. ``predict_proba`` ignores them, but a SHAP explainer built
    on the full booster would not, so the booster is sliced to match.
    """
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    return booster[: best + 1] if best is not None else booster


def shap_explanation(model, X: pd.DataFrame):
    """Compute SHAP values for ``X`` with ``shap.TreeExplainer``.

    Returns a ``shap.Explanation`` with readable feature names (from
    :data:`FEATURE_LABELS`) and readable display values (raw amounts and
    balances instead of their logs) for plots. Uses the booster sliced to
    the best iteration, so values add up to the margin behind
    ``predict_proba``.
    """
    import shap

    raw = shap.TreeExplainer(best_iteration_booster(model))(X)
    return shap.Explanation(
        values=raw.values,
        base_values=raw.base_values,
        data=X.to_numpy(),
        display_data=readable_values(X).to_numpy(),
        feature_names=[FEATURE_LABELS.get(c, c) for c in X.columns],
    )


def check_additivity(model, X: pd.DataFrame, explanation, tol: float = 1e-3) -> float:
    """Assert that base value + SHAP values reproduce the model's scores.

    Returns the largest absolute difference in probability.
    """
    margin = explanation.base_values + explanation.values.sum(axis=1)
    from_shap = 1 / (1 + np.exp(-margin))
    from_model = model.predict_proba(X)[:, 1]
    gap = float(np.max(np.abs(from_shap - from_model)))
    assert gap < tol, f"SHAP values do not add up to the model's scores (max gap {gap:.2e})"
    return gap


def sample_for_shap(y_true, y_pred, n_fraud: int = 500, n_legit: int = 5000,
                    seed: int = 42) -> pd.Index:
    """Pick a small, stratified set of rows to explain.

    Always includes every misclassified row (false positives and false
    negatives, so they can be inspected), then tops up with randomly chosen
    frauds (up to ``n_fraud`` in total) and legitimate rows (up to
    ``n_legit`` in total). Returns index labels in their original order.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    errors = y_true.index[y_true != y_pred]
    chosen = set(errors)

    rng = np.random.default_rng(seed)
    for label, target in [(1, n_fraud), (0, n_legit)]:
        pool = y_true.index[(y_true == label) & ~y_true.index.isin(errors)]
        already = int(((y_true == label) & y_true.index.isin(errors)).sum())
        k = max(0, min(target - already, len(pool)))
        chosen.update(rng.choice(pool, size=k, replace=False))
    return y_true.index[y_true.index.isin(chosen)]


def describe_feature(feature: str, row: pd.Series, shap_value: float) -> str:
    """One plain-English sentence for a feature's contribution to the score.

    ``row`` holds the raw transaction (amount, balances, type, step).
    """
    direction = "raises" if shap_value > 0 else "lowers"
    amount = float(row["amount"])
    orig = float(row["oldbalanceOrg"])
    dest = float(row["oldbalanceDest"])

    if feature == "amount_to_balance_ratio":
        if amount == MAX_PAYSIM_AMOUNT and orig >= MAX_PAYSIM_AMOUNT:
            fact = (f"The transfer is a 10,000,000 chunk of a {orig:,.0f} balance "
                    "(large balances are drained in 10M chunks)")
        elif orig > 0 and abs(amount - orig) < 0.01:
            fact = f"The transfer moves the origin account's entire balance ({amount:,.2f})"
        elif orig == 0:
            fact = "The origin account had no balance before the transfer"
        elif amount > orig:
            fact = f"The amount ({amount:,.0f}) is larger than the origin balance ({orig:,.0f})"
        else:
            fact = f"The transfer uses {100 * amount / orig:.0f}% of the origin balance"
    elif feature == "amount_log":
        fact = (f"The amount is exactly {amount:,.0f}, the largest single transfer"
                if amount == MAX_PAYSIM_AMOUNT else f"The amount is {amount:,.0f}")
    elif feature == "oldbalanceOrg_log":
        fact = ("The origin account was empty" if orig == 0
                else f"The origin account held {orig:,.0f} beforehand")
    elif feature == "oldbalanceDest_log":
        fact = ("The destination account had a zero prior balance" if dest == 0
                else f"The destination account held {dest:,.0f} beforehand")
    elif feature == "dest_new_or_zero":
        fact = ("The destination account had a zero prior balance (new or empty account)"
                if dest == 0 else "The destination account already held funds")
    elif feature == "hour":
        hour = int(row["step"]) % 24
        fact = f"It happened at hour {hour}"
        if hour < 8:
            fact += ", when legitimate activity is low"
    elif feature == "is_transfer":
        fact = f"It is a {row['type']}"
    else:
        fact = f"{FEATURE_LABELS.get(feature, feature)} has its observed value"
    return f"{fact}, which {direction} the fraud score."


@lru_cache(maxsize=1)
def load_headline_model():
    """Load the saved headline model and its validation-chosen threshold.

    Returns ``(model, threshold)``: models/xgb_pre_transaction.joblib and the
    cost-optimal threshold stored in reports/final_metrics.json.
    """
    from src.train import load_model

    model = load_model(HEADLINE_MODEL)
    metrics = json.loads((PROJECT_ROOT / "reports" / "final_metrics.json").read_text())
    return model, float(metrics["thresholds_chosen_on_validation"][HEADLINE_NAME])


def explain_transaction(row, model=None, threshold: float = None, top_n: int = 3) -> dict:
    """Score one transaction and explain it in plain English.

    Parameters
    ----------
    row : dict or pd.Series
        A TRANSFER or CASH_OUT transaction with at least ``step``, ``type``,
        ``amount``, ``oldbalanceOrg`` and ``oldbalanceDest``.
    model, threshold : optional
        Defaults to the saved pre-transaction XGBoost model and the threshold
        chosen on validation in notebook 04.
    top_n : int
        Number of contributing features to describe.

    Returns
    -------
    dict
        ``score`` (fraud probability), ``flagged`` (score >= threshold),
        ``threshold``, ``top_features`` (list of dicts with ``feature``,
        ``label``, ``shap`` and ``text``) and a one-line ``summary``.
    """
    if model is None or threshold is None:
        default_model, default_threshold = load_headline_model()
        model = model if model is not None else default_model
        threshold = threshold if threshold is not None else default_threshold

    tx = pd.DataFrame([dict(row)])
    for col in _POST_TRANSACTION_COLUMNS:   # not used by pre-transaction features
        if col not in tx:
            tx[col] = np.nan
    X = build_features(tx, feature_set="pre_transaction")

    score = float(model.predict_proba(X)[:, 1][0])
    values = shap_explanation(model, X).values[0]
    order = np.argsort(-np.abs(values))[:top_n]

    top = [{"feature": X.columns[i], "label": FEATURE_LABELS.get(X.columns[i], X.columns[i]),
            "shap": float(values[i]), "text": describe_feature(X.columns[i], tx.iloc[0], values[i])}
           for i in order]
    verdict = "Flagged for review" if score >= threshold else "Not flagged"
    return {
        "score": score,
        "flagged": score >= threshold,
        "threshold": threshold,
        "top_features": top,
        "summary": f"{verdict} (score {score:.3f}, threshold {threshold:.3f}). {top[0]['text']}",
    }
