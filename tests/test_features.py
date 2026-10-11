"""Tests for src.features: splitting, feature values and leakage safety."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    BASE_FEATURES,
    BEHAVIOR_FEATURES,
    FORBIDDEN_COLUMNS,
    add_behavior_features,
    build_features,
    default_boundaries,
    filter_relevant,
    time_split,
)

COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
           "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]


def make_df(rows):
    """Build a PaySim-shaped DataFrame from a list of row lists."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["type"] = df["type"].astype("category")
    return df


@pytest.fixture
def tiny():
    """Four hand-made rows with easy-to-check values."""
    return make_df([
        # step type        amount  orig  oldOrg newOrg dest  oldDst newDst fraud flag
        [1,  "TRANSFER",  100.0, "C1", 100.0,   0.0, "C9",   0.0,   0.0, 1, 0],
        [2,  "CASH_OUT",   50.0, "C2", 200.0, 150.0, "C8", 500.0, 550.0, 0, 0],
        [25, "TRANSFER",   30.0, "C3",   0.0,   0.0, "C7",   0.0,  30.0, 0, 0],
        [26, "PAYMENT",    10.0, "C4",  50.0,  40.0, "M1",   0.0,   0.0, 0, 0],
    ])


@pytest.fixture
def timeline():
    """One row per step for steps 1-100, a fraud every 10th step."""
    rows = [[s, "TRANSFER" if s % 2 else "CASH_OUT", 10.0, f"C{s}", 10.0, 0.0,
             f"C{s + 1000}", 0.0, 10.0, int(s % 10 == 0), 0] for s in range(1, 101)]
    return make_df(rows)


# ---------------------------------------------------------------- filtering

def test_filter_relevant_keeps_only_transfer_and_cash_out(tiny):
    out = filter_relevant(tiny, verbose=False)
    assert set(out["type"].astype(str)) == {"TRANSFER", "CASH_OUT"}
    assert len(out) == 3
    assert out["isFraud"].sum() == tiny["isFraud"].sum()


# ---------------------------------------------------------------- splitting

def test_default_boundaries_paysim_range():
    assert default_boundaries(pd.Series([1, 743])) == (520, 632)


def test_time_split_no_step_overlap(timeline):
    train_end, val_end = default_boundaries(timeline["step"])
    train, val, test = time_split(timeline, train_end, val_end, verbose=False)

    train_steps, val_steps, test_steps = (set(p["step"]) for p in (train, val, test))
    assert not (train_steps & val_steps)
    assert not (train_steps & test_steps)
    assert not (val_steps & test_steps)
    # Strict time order: everything in val is after train, test after val.
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()
    assert len(train) + len(val) + len(test) == len(timeline)
    assert (len(train), len(val), len(test)) == (70, 15, 15)


def test_time_split_rejects_bad_boundaries(timeline):
    with pytest.raises(ValueError):
        time_split(timeline, 80, 70, verbose=False)


# ---------------------------------------------------------------- features

def test_forbidden_columns_absent(tiny):
    rel = filter_relevant(tiny, verbose=False)
    for include_behavior in (False, True):
        X = build_features(rel, include_behavior=include_behavior)
        assert not set(FORBIDDEN_COLUMNS) & set(X.columns)
    assert list(build_features(rel).columns) == BASE_FEATURES
    assert list(build_features(rel, include_behavior=True).columns) == BASE_FEATURES + BEHAVIOR_FEATURES


def test_feature_values_on_hand_made_rows(tiny):
    rel = filter_relevant(tiny, verbose=False)
    X = build_features(rel)

    # Row 0: TRANSFER 100 that empties a 100 balance into an empty account.
    r = X.loc[0]
    assert r["errorBalanceOrig"] == 0.0           # 0 + 100 - 100
    assert r["errorBalanceDest"] == 100.0         # 0 + 100 - 0
    assert r["orig_emptied"] == 1
    assert r["dest_new_or_zero"] == 1
    assert r["hour"] == 1
    assert r["amount_to_balance_ratio"] == pytest.approx(100 / 101)
    assert r["is_transfer"] == 1
    assert r["amount_log"] == pytest.approx(np.log1p(100))

    # Row 1: ordinary CASH_OUT with consistent balances.
    r = X.loc[1]
    assert r["errorBalanceOrig"] == 0.0           # 150 + 50 - 200
    assert r["errorBalanceDest"] == 0.0           # 500 + 50 - 550
    assert r["orig_emptied"] == 0
    assert r["dest_new_or_zero"] == 0
    assert r["is_transfer"] == 0

    # Row 2: origin already at 0, so it is NOT "emptied"; step 25 -> hour 1.
    r = X.loc[2]
    assert r["errorBalanceOrig"] == 30.0          # 0 + 30 - 0
    assert r["orig_emptied"] == 0
    assert r["hour"] == 1
    assert r["amount_to_balance_ratio"] == pytest.approx(30.0)


def test_features_keep_index_aligned_with_labels(tiny):
    rel = filter_relevant(tiny, verbose=False)
    X = build_features(rel)
    assert X.index.equals(rel.index)


def test_build_features_requires_filtered_data(tiny):
    with pytest.raises(ValueError, match="filter_relevant"):
        build_features(tiny)


# ---------------------------------------------------------------- behaviour features

@pytest.fixture
def history():
    """Account A transacts in steps 1, 1, 3, 5; account B once in step 3."""
    return make_df([
        [1, "TRANSFER", 100.0, "A", 1e3, 900.0, "X", 0.0, 0.0, 0, 0],
        [1, "CASH_OUT", 300.0, "A", 900.0, 600.0, "Y", 0.0, 0.0, 0, 0],
        [3, "TRANSFER",  50.0, "A", 600.0, 550.0, "X", 0.0, 0.0, 0, 0],
        [3, "CASH_OUT",  70.0, "B", 70.0,   0.0, "Z", 0.0, 0.0, 1, 0],
        [5, "TRANSFER", 999.0, "A", 550.0,   0.0, "Y", 0.0, 0.0, 1, 0],
    ])


def test_behavior_features_use_only_earlier_steps(history):
    out = add_behavior_features(history)
    # Step 1 rows: no history. They share a step, so they must not see each other.
    assert out.loc[[0, 1], "prev_txn_count"].tolist() == [0, 0]
    assert out.loc[[0, 1], "prev_mean_amount"].tolist() == [0.0, 0.0]
    # Step 3, account A: sees the two step-1 rows only.
    assert out.loc[2, "prev_txn_count"] == 2
    assert out.loc[2, "prev_mean_amount"] == pytest.approx(200.0)   # (100 + 300) / 2
    # Account B: first appearance, A's history does not leak across accounts.
    assert out.loc[3, "prev_txn_count"] == 0
    # Step 5, account A: three earlier rows, its own 999 excluded.
    assert out.loc[4, "prev_txn_count"] == 3
    assert out.loc[4, "prev_mean_amount"] == pytest.approx(150.0)   # (100 + 300 + 50) / 3


def test_behavior_features_never_use_current_or_future_rows(history):
    base = add_behavior_features(history)

    # Changing a row's own amount must not change its own behaviour features.
    changed = history.copy()
    changed.loc[4, "amount"] = 1e9
    assert base.loc[4, BEHAVIOR_FEATURES].equals(add_behavior_features(changed).loc[4, BEHAVIOR_FEATURES])

    # Changing a future row must not change any earlier row's features.
    changed = history.copy()
    changed.loc[4, "amount"] = 1e9
    after = add_behavior_features(changed)
    assert base.loc[:3, BEHAVIOR_FEATURES].equals(after.loc[:3, BEHAVIOR_FEATURES])

    # Labels are never used: flipping isFraud changes nothing.
    flipped = history.assign(isFraud=1 - history["isFraud"])
    assert base[BEHAVIOR_FEATURES].equals(add_behavior_features(flipped)[BEHAVIOR_FEATURES])


def test_behavior_features_independent_of_row_order(history):
    shuffled = history.sample(frac=1, random_state=42)
    out = add_behavior_features(shuffled).sort_index()
    assert out[BEHAVIOR_FEATURES].equals(add_behavior_features(history)[BEHAVIOR_FEATURES])


def test_build_features_reuses_precomputed_behavior(history):
    full = add_behavior_features(history)
    rel = full[full["step"] >= 3]  # history from step 1 must survive the slice
    X = build_features(rel, include_behavior=True)
    assert X.loc[4, "prev_txn_count"] == 3


# ---------------------------------------------------------------- pre-transaction feature set

def test_pre_transaction_features_ignore_post_transaction_balances(tiny):
    """Changing the balances recorded AFTER the payment must not change any feature."""
    from src.features import PRE_TRANSACTION_FEATURES

    rel = filter_relevant(tiny, verbose=False)
    before = build_features(rel, feature_set="pre_transaction")
    assert list(before.columns) == PRE_TRANSACTION_FEATURES

    changed = rel.assign(newbalanceOrig=rel["newbalanceOrig"] + 12345.0,
                         newbalanceDest=rel["newbalanceDest"] * 7 + 1)
    after = build_features(changed, feature_set="pre_transaction")
    pd.testing.assert_frame_equal(before, after)


def test_pre_transaction_feature_values(tiny):
    rel = filter_relevant(tiny, verbose=False)
    X = build_features(rel, feature_set="pre_transaction")
    assert X.loc[0, "oldbalanceOrg_log"] == pytest.approx(np.log1p(100.0))
    assert X.loc[1, "oldbalanceDest_log"] == pytest.approx(np.log1p(500.0))
    assert X.loc[2, "oldbalanceOrg_log"] == 0.0
    assert not set(FORBIDDEN_COLUMNS) & set(X.columns)


def test_unknown_feature_set_rejected(tiny):
    rel = filter_relevant(tiny, verbose=False)
    with pytest.raises(ValueError, match="feature_set"):
        build_features(rel, feature_set="everything")
