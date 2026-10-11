"""Row filtering, time-based splitting and feature engineering for PaySim.

Typical use::

    df = load_data(money_dtype="float64")
    df = add_behavior_features(df)          # optional, on the FULL history
    df = filter_relevant(df)
    train_end, val_end = default_boundaries(df["step"])
    train, val, test = time_split(df, train_end, val_end)
    X_train, y_train = build_features(train, include_behavior=True), train["isFraud"]
"""

import numpy as np
import pandas as pd

RELEVANT_TYPES = ["TRANSFER", "CASH_OUT"]

# Columns that must never reach a model.
FORBIDDEN_COLUMNS = ["isFlaggedFraud", "nameOrig", "nameDest", "step", "isFraud"]

BASE_FEATURES = [
    "amount_log",
    "errorBalanceOrig",
    "errorBalanceDest",
    "orig_emptied",
    "dest_new_or_zero",
    "hour",
    "amount_to_balance_ratio",
    "is_transfer",
]
# Only information known before the payment is processed: no column here
# uses newbalanceOrig or newbalanceDest.
PRE_TRANSACTION_FEATURES = [
    "amount_log",
    "oldbalanceOrg_log",
    "oldbalanceDest_log",
    "amount_to_balance_ratio",
    "dest_new_or_zero",
    "hour",
    "is_transfer",
]
FEATURE_SETS = {"full": BASE_FEATURES, "pre_transaction": PRE_TRANSACTION_FEATURES}
BEHAVIOR_FEATURES = ["prev_txn_count", "prev_mean_amount"]


def _fraud_summary(df: pd.DataFrame) -> str:
    """One-line row count, fraud count and fraud rate."""
    n, n_fraud = len(df), int(df["isFraud"].sum())
    rate = 100 * n_fraud / n if n else 0.0
    return f"{n:>10,} rows | {n_fraud:>6,} fraud | fraud rate {rate:.4f}%"


def filter_relevant(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep only TRANSFER and CASH_OUT rows, the only types that contain fraud.

    The EDA showed PAYMENT, CASH_IN and DEBIT have zero fraud, so dropping
    them loses no positives and raises the base rate.
    """
    out = df[df["type"].isin(RELEVANT_TYPES)].copy()
    if isinstance(out["type"].dtype, pd.CategoricalDtype):
        out["type"] = out["type"].cat.remove_unused_categories()
    if verbose:
        print(f"Before filter: {_fraud_summary(df)}")
        print(f"After filter:  {_fraud_summary(out)}")
        print(f"Removed {len(df) - len(out):,} rows "
              f"({100 * (len(df) - len(out)) / len(df):.1f}%), "
              f"{int(df['isFraud'].sum() - out['isFraud'].sum())} of them fraud.")
    return out


def default_boundaries(steps: pd.Series, train_frac: float = 0.70,
                       val_frac: float = 0.15) -> tuple:
    """Return (train_end, val_end) so the step range splits ~70/15/15.

    Boundaries are by number of steps (hours), not number of rows. For PaySim
    (steps 1 to 743) this gives train = 1-520, val = 521-632, test = 633-743.
    """
    first, last = int(steps.min()), int(steps.max())
    n_steps = last - first + 1
    train_end = first - 1 + round(train_frac * n_steps)
    val_end = first - 1 + round((train_frac + val_frac) * n_steps)
    return train_end, val_end


def time_split(df: pd.DataFrame, train_end: int, val_end: int,
               verbose: bool = True) -> tuple:
    """Split rows by ``step`` into train / validation / test.

    - train: step <= train_end
    - val:   train_end < step <= val_end
    - test:  step > val_end

    Every row lands in exactly one split, and no step appears in two splits,
    so the model is always evaluated on hours that come after its training data.
    """
    if not train_end < val_end:
        raise ValueError(f"Need train_end < val_end, got {train_end} and {val_end}.")

    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]

    train_steps, val_steps, test_steps = (set(s["step"].unique()) for s in (train, val, test))
    assert not (train_steps & val_steps), "train and val share steps"
    assert not (train_steps & test_steps), "train and test share steps"
    assert not (val_steps & test_steps), "val and test share steps"
    assert len(train) + len(val) + len(test) == len(df), "rows lost in split"

    if verbose:
        for name, part, lo, hi in [
            ("train", train, int(df["step"].min()), train_end),
            ("val", val, train_end + 1, val_end),
            ("test", test, val_end + 1, int(df["step"].max())),
        ]:
            print(f"{name:<5} steps {lo:>3}-{hi:<3} {_fraud_summary(part)}")
    return train, val, test


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-only behaviour features for each origin account (``nameOrig``).

    - ``prev_txn_count``: how many transactions this account made in
      earlier steps.
    - ``prev_mean_amount``: mean amount of those earlier transactions
      (0 when there are none).

    Leakage reasoning:

    - Only **strictly earlier steps** are used. Transactions in the same
      hour are excluded because their order within a step is not reliable,
      and including them could let a row "see" one that happened after it.
    - The current row is never included: per-step totals are cumulated per
      account and the current step's own total is subtracted (a shifted
      cumulative sum).
    - The label (``isFraud``) is never used.
    - Because every value depends only on the past, it is safe to compute
      this once on the full timeline before splitting. A validation row may
      use history from the training period, which is exactly what a live
      system would know.

    Call this on the **unfiltered** data so history includes all transaction
    types, then filter. Note that in PaySim origin IDs almost never repeat
    (99.7% appear once), so these features are nearly always 0. They are
    included to show the leakage-safe pattern, not because they add signal.
    """
    amounts = pd.DataFrame(
        {"nameOrig": df["nameOrig"], "step": df["step"],
         "amount": df["amount"].astype("float64")}
    )
    per_step = (
        amounts.groupby(["nameOrig", "step"], observed=True)["amount"]
        .agg(n="size", total="sum")
        .sort_index()
    )
    by_account = per_step.groupby(level="nameOrig", observed=True)
    prev_count = by_account["n"].cumsum() - per_step["n"]
    prev_total = by_account["total"].cumsum() - per_step["total"]
    prev_mean = (prev_total / prev_count.where(prev_count > 0)).fillna(0.0)

    keys = pd.MultiIndex.from_arrays([df["nameOrig"], df["step"]])
    out = df.copy()
    out["prev_txn_count"] = prev_count.reindex(keys).to_numpy().astype("int32")
    out["prev_mean_amount"] = prev_mean.reindex(keys).to_numpy()
    return out


def build_features(df: pd.DataFrame, include_behavior: bool = False,
                   feature_set: str = "full") -> pd.DataFrame:
    """Return the model-ready feature matrix for TRANSFER / CASH_OUT rows.

    ``feature_set`` chooses the columns:

    - ``"full"``: the features listed below, including the balance errors,
      which use balances recorded *after* the transaction.
    - ``"pre_transaction"``: only information available before the payment
      is processed: ``amount_log``, ``oldbalanceOrg_log``,
      ``oldbalanceDest_log`` (log1p of the opening balances),
      ``amount_to_balance_ratio``, ``dest_new_or_zero``, ``hour`` and
      ``is_transfer``. In PaySim the post-transaction balances encode how
      the simulator books fraud (fraud origin balances add up exactly; fraud
      destinations are often never credited), which makes the "full" set
      close to perfect but unrealistic. This set is the realistic one.

    The output keeps ``df``'s index so it lines up with ``df["isFraud"]``.
    Money values are converted to float64 before arithmetic so balance errors
    are not swamped by float32 rounding.

    Features:

    - ``amount_log``: log1p(amount).
    - ``errorBalanceOrig``: newbalanceOrig + amount - oldbalanceOrg.
    - ``errorBalanceDest``: oldbalanceDest + amount - newbalanceDest.
    - ``orig_emptied``: 1 if the origin went from a positive balance to 0.
    - ``dest_new_or_zero``: 1 if the destination balance was 0 before.
    - ``hour``: step % 24. Predictive in PaySim only because legitimate
      volume drops at night (see EDA); treat it with caution.
    - ``amount_to_balance_ratio``: amount / (oldbalanceOrg + 1).
    - ``is_transfer``: 1 for TRANSFER, 0 for CASH_OUT.
    - with ``include_behavior=True``: ``prev_txn_count`` and
      ``prev_mean_amount`` (see :func:`add_behavior_features`). If they are
      already in ``df`` (computed earlier on the full history) they are
      reused; otherwise they are computed from ``df`` alone.

    ``isFlaggedFraud``, ``nameOrig``, ``nameDest``, ``step`` and the label
    are never included.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {list(FEATURE_SETS)}.")
    other_types = set(df["type"].astype(str).unique()) - set(RELEVANT_TYPES)
    if other_types:
        raise ValueError(f"Run filter_relevant first; found types {sorted(other_types)}.")

    amount = df["amount"].astype("float64")
    old_orig = df["oldbalanceOrg"].astype("float64")
    new_orig = df["newbalanceOrig"].astype("float64")
    old_dest = df["oldbalanceDest"].astype("float64")
    new_dest = df["newbalanceDest"].astype("float64")

    X = pd.DataFrame(index=df.index)
    X["amount_log"] = np.log1p(amount)
    X["errorBalanceOrig"] = new_orig + amount - old_orig
    X["errorBalanceDest"] = old_dest + amount - new_dest
    X["orig_emptied"] = ((new_orig == 0) & (old_orig > 0)).astype("int8")
    X["dest_new_or_zero"] = (old_dest == 0).astype("int8")
    X["hour"] = (df["step"] % 24).astype("int8")
    X["amount_to_balance_ratio"] = amount / (old_orig + 1)
    X["is_transfer"] = (df["type"].astype(str) == "TRANSFER").astype("int8")
    X["oldbalanceOrg_log"] = np.log1p(old_orig)
    X["oldbalanceDest_log"] = np.log1p(old_dest)
    X = X[FEATURE_SETS[feature_set]].copy()

    if include_behavior:
        if not set(BEHAVIOR_FEATURES).issubset(df.columns):
            df = add_behavior_features(df)
        X[BEHAVIOR_FEATURES] = df[BEHAVIOR_FEATURES]

    leaked = set(FORBIDDEN_COLUMNS) & set(X.columns)
    assert not leaked, f"Forbidden columns in features: {leaked}"
    return X
