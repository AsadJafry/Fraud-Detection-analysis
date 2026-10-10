"""Summary tables for exploratory analysis of PaySim.

Each function takes the loaded DataFrame and returns a small pandas object
that the EDA notebook displays or plots.
"""

import pandas as pd

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]


def _fraud_rate_table(df: pd.DataFrame, key) -> pd.DataFrame:
    """Transactions, fraud count and fraud rate (%) grouped by ``key``.

    ``isFraud`` is cast to int64 first: it is stored as int8 to save memory,
    and int8 sums overflow on groups with more than 127 frauds.
    """
    out = (
        df["isFraud"]
        .astype("int64")
        .groupby(key, observed=True)
        .agg(n_txn="size", n_fraud="sum")
    )
    out["fraud_rate_pct"] = 100 * out["n_fraud"] / out["n_txn"]
    return out


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missing count and memory use (MB)."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "memory_mb": (df.memory_usage(deep=True, index=False) / 1e6).round(1),
        }
    )


def count_duplicates(df: pd.DataFrame) -> int:
    """Number of fully duplicated rows."""
    return int(df.duplicated().sum())


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of legitimate vs fraudulent transactions."""
    counts = df["isFraud"].astype("int64").value_counts().sort_index()
    out = pd.DataFrame({"count": counts, "pct": 100 * counts / counts.sum()})
    out.index = out.index.map({0: "legit", 1: "fraud"})
    out.index.name = "class"
    return out


def fraud_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count, fraud count and fraud rate (%) per transaction type."""
    out = _fraud_rate_table(df, df["type"])
    return out.sort_values("n_txn", ascending=False)


def fraud_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions, fraud count and fraud rate (%) for every hourly step.

    Adds ``day`` (1-based) and ``hour`` (step % 24) for convenience.
    """
    out = _fraud_rate_table(df, df["step"])
    out["day"] = (out.index - 1) // 24 + 1
    out["hour"] = out.index % 24
    return out


def fraud_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions, fraud count and fraud rate (%) by hour of day (step % 24)."""
    return _fraud_rate_table(df, (df["step"] % 24).rename("hour"))


def fraud_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions, fraud count and fraud rate (%) per simulated day."""
    return _fraud_rate_table(df, ((df["step"] - 1) // 24 + 1).rename("day"))


def balance_patterns(df: pd.DataFrame, types=FRAUD_TYPES) -> pd.DataFrame:
    """Share (%) of transactions with telling balance patterns, by class.

    Restricted to ``types`` (TRANSFER and CASH_OUT by default) so that fraud is
    compared with legitimate transactions of the same kind.

    Patterns:
    - ``orig_emptied``: origin balance is 0 after the transaction.
    - ``orig_zero_before``: origin balance was already 0 before.
    - ``orig_drained``: origin went from a positive balance to exactly 0.
    - ``dest_zero_before``: destination balance was 0 before the transaction.
    - ``emptied_and_dest_zero``: origin emptied and destination was at 0.
    """
    sub = df[df["type"].isin(types)]
    flags = pd.DataFrame(
        {
            "orig_emptied": sub["newbalanceOrig"] == 0,
            "orig_zero_before": sub["oldbalanceOrg"] == 0,
            "dest_zero_before": sub["oldbalanceDest"] == 0,
        }
    )
    flags["orig_drained"] = flags["orig_emptied"] & ~flags["orig_zero_before"]
    flags["emptied_and_dest_zero"] = flags["orig_emptied"] & flags["dest_zero_before"]
    flags = flags[
        ["orig_emptied", "orig_zero_before", "orig_drained",
         "dest_zero_before", "emptied_and_dest_zero"]
    ]
    out = 100 * flags.groupby(sub["isFraud"]).mean()
    out.index = out.index.map({0: "legit", 1: "fraud"})
    out.index.name = "class"
    return out


def account_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """How often origin and destination account IDs repeat.

    Returns one row per ID column with the number of unique IDs, the share of
    rows whose ID appears only once, and the maximum number of appearances.
    A ``nameDest (fraud only)`` row shows reuse among fraud destinations.
    """
    rows = {}
    for label, ids in [
        ("nameOrig", df["nameOrig"]),
        ("nameDest", df["nameDest"]),
        ("nameDest (fraud only)", df.loc[df["isFraud"] == 1, "nameDest"]),
    ]:
        counts = ids.value_counts()
        rows[label] = {
            "n_rows": len(ids),
            "n_unique": len(counts),
            "pct_rows_id_seen_once": 100 * (counts == 1).sum() / len(ids),
            "max_appearances": int(counts.max()),
        }
    return pd.DataFrame(rows).T


def id_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of account-ID prefixes (C = customer, M = merchant) by class."""
    out = pd.DataFrame(
        {
            "orig_prefix": df["nameOrig"].str[0],
            "dest_prefix": df["nameDest"].str[0],
            "isFraud": df["isFraud"],
        }
    )
    return out.groupby(["isFraud", "orig_prefix", "dest_prefix"]).size().rename("count")


def flagged_vs_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of the legacy rule flag against the true label."""
    return pd.crosstab(
        df["isFlaggedFraud"].map({0: "not flagged", 1: "flagged"}),
        df["isFraud"].map({0: "legit", 1: "fraud"}),
        margins=True,
    )
