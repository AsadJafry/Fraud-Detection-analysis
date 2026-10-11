"""Loading and validating the PaySim transaction data."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

MONEY_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _string_dtype() -> str:
    """Return a compact string dtype: pyarrow-backed if available, else object."""
    try:
        import pyarrow  # noqa: F401

        return "string[pyarrow]"
    except ImportError:
        return "object"


def get_dtypes(money_dtype: str = "float32") -> dict:
    """Return the memory-efficient dtype for every PaySim column.

    - ``step`` fits in int16 (max 743).
    - Money columns default to float32. This halves memory but keeps only ~7
      significant digits, so very large balances lose sub-unit precision.
      Pass ``money_dtype="float64"`` for modelling, where exact balance
      arithmetic (e.g. balance-error features) matters.
    - ``type`` is a category with the five known transaction types.
    - Account IDs use pyarrow strings when pyarrow is installed (much smaller
      than Python objects).
    """
    dtypes = {col: money_dtype for col in MONEY_COLUMNS}
    dtypes.update(
        {
            "step": "int16",
            "type": pd.CategoricalDtype(TRANSACTION_TYPES),
            "nameOrig": _string_dtype(),
            "nameDest": _string_dtype(),
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        }
    )
    return dtypes


def validate_columns(columns) -> None:
    """Raise ValueError if the columns differ from the expected PaySim schema."""
    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
    unexpected = [c for c in columns if c not in EXPECTED_COLUMNS]
    if missing or unexpected:
        raise ValueError(
            f"Unexpected PaySim schema. Missing: {missing}. Unexpected: {unexpected}."
        )


def validate_data(df: pd.DataFrame) -> None:
    """Basic sanity checks on loaded data. Raises ValueError on failure."""
    validate_columns(list(df.columns))
    if df["type"].isna().any():
        raise ValueError("Found transaction types outside the expected set.")
    for col in ["isFraud", "isFlaggedFraud"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Column {col} must contain only 0 and 1.")
    if (df["step"] < 1).any():
        raise ValueError("Column step must be >= 1.")


def load_data(path=DEFAULT_DATA_PATH, nrows=None, money_dtype: str = "float32") -> pd.DataFrame:
    """Load the PaySim CSV with compact dtypes and validate it.

    Parameters
    ----------
    path : str or Path
        Path to the PaySim CSV. Defaults to ``data/paysim.csv``.
    nrows : int, optional
        Read only the first ``nrows`` rows (handy for quick tests).
    money_dtype : str
        dtype for amount and balance columns: "float32" (default, compact)
        or "float64" (exact enough for balance arithmetic).

    Returns
    -------
    pd.DataFrame
        The validated transactions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    validate_columns(header)

    # Read `type` as plain strings first so unknown values can be reported
    # clearly, then convert it to the fixed categorical dtype.
    dtypes = get_dtypes(money_dtype)
    type_dtype = dtypes.pop("type")
    df = pd.read_csv(path, dtype={**dtypes, "type": "str"}, nrows=nrows)

    unknown = set(df["type"].unique()) - set(TRANSACTION_TYPES)
    if unknown:
        raise ValueError(f"Found transaction types outside the expected set: {sorted(unknown)}")
    df["type"] = df["type"].astype(type_dtype)

    validate_data(df)
    return df
