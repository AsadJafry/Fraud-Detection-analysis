"""Tests for src.models."""

import pandas as pd

from src.models import full_balance_rule


def test_full_balance_rule():
    df = pd.DataFrame({
        "amount":        [100.0, 100.0, 50.0, 10_000_000.0, 10_000_000.0, 0.0],
        "oldbalanceOrg": [100.0, 100.004, 100.0, 25_000_000.0, 5_000_000.0, 0.0],
    })
    # exact drain, drain within tolerance, partial, capped chunk, 10M > balance, empty account
    assert full_balance_rule(df).tolist() == [1, 1, 0, 1, 0, 0]
