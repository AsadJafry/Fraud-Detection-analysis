"""Tests for src.eda."""

import pandas as pd

from src import eda


def test_fraud_rate_no_int8_overflow():
    # 200 frauds in one group would overflow an int8 sum (max 127).
    df = pd.DataFrame(
        {
            "type": pd.Categorical(["TRANSFER"] * 400),
            "step": [1] * 400,
            "isFraud": pd.array([1] * 200 + [0] * 200, dtype="int8"),
        }
    )
    out = eda.fraud_by_type(df)
    assert out.loc["TRANSFER", "n_fraud"] == 200
    assert out.loc["TRANSFER", "fraud_rate_pct"] == 50.0
    hour = eda.fraud_by_hour(df)
    assert hour.loc[1, "fraud_rate_pct"] == 50.0
