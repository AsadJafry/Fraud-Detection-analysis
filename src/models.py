"""Model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def make_logreg_baseline() -> Pipeline:
    """Scaled logistic regression with balanced class weights.

    The scaler sits inside the pipeline, so calling ``fit`` on training data
    learns the means and standard deviations from training rows only.
    ``class_weight="balanced"`` up-weights the rare fraud class so the model
    does not just predict "legit" everywhere.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                         random_state=SEED)),
        ]
    )


MAX_PAYSIM_AMOUNT = 10_000_000  # PaySim splits larger transfers into 10M chunks


def full_balance_rule(df, tol: float = 0.01):
    """One-line rule: flag a transaction that moves the origin's ENTIRE balance.

    Returns 1 when the amount equals the full opening balance of the origin
    account (within ``tol``), or when it is a capped 10M chunk taken from an
    account holding at least 10M; otherwise 0.

    In PaySim this is how fraud is generated: the fraudster always empties
    the account. The rule uses only pre-transaction information (amount and
    opening balance) and no labels, yet it matches XGBoost, which shows how
    little is left for machine learning to learn on this dataset.
    """
    amount = df["amount"].astype("float64")
    old = df["oldbalanceOrg"].astype("float64")
    exact_drain = ((amount - old).abs() < tol) & (old > 0)
    capped_chunk = (amount == MAX_PAYSIM_AMOUNT) & (old >= MAX_PAYSIM_AMOUNT)
    return (exact_drain | capped_chunk).astype("int8").to_numpy()
