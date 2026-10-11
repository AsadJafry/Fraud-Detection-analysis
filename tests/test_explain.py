"""Tests for src.explain."""

import numpy as np
import pandas as pd
import pytest

from src.explain import describe_feature, sample_for_shap

ROW = pd.Series({"step": 28, "type": "TRANSFER", "amount": 181.0,
                 "oldbalanceOrg": 181.0, "oldbalanceDest": 0.0})


# ---------------------------------------------------------------- sampling

def test_sample_includes_all_errors_and_respects_sizes():
    y_true = pd.Series([1] * 50 + [0] * 1000, index=range(100, 1150))
    y_pred = y_true.copy()
    y_pred.iloc[[0, 1]] = 0          # two false negatives
    y_pred.iloc[[60, 70, 80]] = 1    # three false positives
    idx = sample_for_shap(y_true, y_pred, n_fraud=20, n_legit=100, seed=42)

    errors = set(y_true.index[y_true != y_pred])
    assert errors <= set(idx)
    assert (y_true[idx] == 1).sum() == 20
    assert (y_true[idx] == 0).sum() == 100
    assert idx.is_monotonic_increasing


def test_sample_takes_everything_when_class_is_small():
    y = pd.Series([1, 1, 0, 0, 0])
    idx = sample_for_shap(y, y, n_fraud=500, n_legit=5000)
    assert len(idx) == 5


# ---------------------------------------------------------------- wording

def test_describe_full_balance_transfer():
    text = describe_feature("amount_to_balance_ratio", ROW, 2.5)
    assert "entire balance" in text and "raises" in text


def test_describe_empty_origin_and_direction():
    row = ROW.copy()
    row["oldbalanceOrg"] = 0.0
    text = describe_feature("amount_to_balance_ratio", row, -1.0)
    assert "no balance" in text and "lowers" in text


def test_describe_destination_and_hour():
    assert "zero prior balance" in describe_feature("dest_new_or_zero", ROW, 0.3)
    assert "hour 4" in describe_feature("hour", ROW, 0.1)          # 28 % 24 = 4
    assert "TRANSFER" in describe_feature("is_transfer", ROW, 0.1)


# ---------------------------------------------------------------- with real xgboost + shap

@pytest.fixture
def small_model():
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    from src.features import build_features
    from src.train import make_xgb

    rng = np.random.default_rng(42)

    def make(n, n_fraud, steps):
        old = rng.uniform(1_000, 100_000, n)
        amount = old * rng.uniform(0.05, 0.9, n)
        amount[:n_fraud] = old[:n_fraud]                     # fraud empties the account
        return pd.DataFrame({
            "step": rng.integers(*steps, n), "type": rng.choice(["TRANSFER", "CASH_OUT"], n),
            "amount": amount, "oldbalanceOrg": old, "newbalanceOrig": old - amount,
            "oldbalanceDest": rng.choice([0.0, 5_000.0], n), "newbalanceDest": 0.0,
            "isFraud": np.r_[np.ones(n_fraud), np.zeros(n - n_fraud)].astype(int),
        })

    train, val = make(3000, 60, (1, 400)), make(1000, 20, (400, 500))
    X_tr = build_features(train, feature_set="pre_transaction")
    X_va = build_features(val, feature_set="pre_transaction")
    model = make_xgb({"n_estimators": 200}).fit(X_tr, train["isFraud"],
                                                eval_set=[(X_va, val["isFraud"])], verbose=False)
    return model, X_va, val


def test_shap_values_add_up_to_model_scores(small_model):
    from src.explain import check_additivity, shap_explanation

    model, X_va, _ = small_model
    assert model.best_iteration < 199        # early stopping left unused trees behind
    explanation = shap_explanation(model, X_va)
    assert check_additivity(model, X_va, explanation) < 1e-3


def test_explain_transaction_output(small_model):
    from src.explain import explain_transaction

    model, _, val = small_model
    result = explain_transaction(val.iloc[0], model=model, threshold=0.5)
    assert 0 <= result["score"] <= 1
    assert result["flagged"] == (result["score"] >= 0.5)
    assert len(result["top_features"]) == 3
    assert all(f["text"].endswith("fraud score.") for f in result["top_features"])
    assert result["summary"].startswith(("Flagged", "Not flagged"))


def test_readable_values_undo_log():
    from src.explain import readable_values

    X = pd.DataFrame({"amount_log": [np.log1p(181.0)], "hour": [4]})
    out = readable_values(X)
    assert out.loc[0, "amount_log"] == 181.0 and out.loc[0, "hour"] == 4


def test_describe_10m_chunk():
    row = pd.Series({"step": 702, "type": "TRANSFER", "amount": 10_000_000.0,
                     "oldbalanceOrg": 33_171_085.59, "oldbalanceDest": 0.0})
    assert "10M chunks" in describe_feature("amount_to_balance_ratio", row, 1.1)
    assert "largest single transfer" in describe_feature("amount_log", row, 1.7)
