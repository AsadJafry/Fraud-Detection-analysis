"""Tests for src.data."""

import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, load_data

ROWS = [
    [1, "PAYMENT", 9839.64, "C1", 170136.0, 160296.36, "M1", 0.0, 0.0, 0, 0],
    [1, "TRANSFER", 181.0, "C2", 181.0, 0.0, "C3", 0.0, 0.0, 1, 0],
    [2, "CASH_OUT", 181.0, "C4", 181.0, 0.0, "C5", 21182.0, 0.0, 1, 0],
]


def _write_csv(tmp_path, rows=ROWS, columns=EXPECTED_COLUMNS):
    path = tmp_path / "paysim.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_load_data_dtypes(tmp_path):
    df = load_data(_write_csv(tmp_path))
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df["step"].dtype == "int16"
    assert df["amount"].dtype == "float32"
    assert isinstance(df["type"].dtype, pd.CategoricalDtype)
    assert df["isFraud"].sum() == 2


def test_load_data_rejects_missing_column(tmp_path):
    cols = EXPECTED_COLUMNS[:-1]
    path = _write_csv(tmp_path, rows=[r[:-1] for r in ROWS], columns=cols)
    with pytest.raises(ValueError, match="isFlaggedFraud"):
        load_data(path)


def test_load_data_rejects_unknown_type(tmp_path):
    rows = [ROWS[0][:1] + ["WIRE"] + ROWS[0][2:]]
    with pytest.raises(ValueError, match="transaction types"):
        load_data(_write_csv(tmp_path, rows=rows))


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")
