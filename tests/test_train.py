"""Tests for src.train (run against the real xgboost and imbalanced-learn)."""

import numpy as np
import pandas as pd
import pytest

from src import train
from src.train import (
    STRATEGIES,
    check_xgboost_version,
    random_search,
    resample,
    scale_pos_weight_value,
    train_xgb,
)


@pytest.fixture
def data():
    """Small separable dataset: fraud has a high value in feature a."""
    rng = np.random.default_rng(42)

    def make(n, n_pos):
        y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(int)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n) + 3 * y,
            "b": rng.normal(0, 1, n),
            "flag": (rng.random(n) < 0.5).astype("int8"),
        })
        return X, pd.Series(y, name="isFraud")

    X_tr, y_tr = make(3000, 60)
    X_va, y_va = make(1000, 20)
    return X_tr, y_tr, X_va, y_va


def test_xgboost_version_supports_constructor_early_stopping():
    assert int(check_xgboost_version().split(".")[0]) >= 2


def test_scale_pos_weight_value():
    assert scale_pos_weight_value([0, 0, 0, 1]) == 3.0
    with pytest.raises(ValueError):
        scale_pos_weight_value([0, 0])


def test_resample_ratios(data):
    X, y, _, _ = data
    X_s, y_s = resample(X, y, "smote", ratio=0.1)
    assert y_s.sum() == pytest.approx(0.1 * (y_s == 0).sum(), abs=1)
    assert (y_s == 0).sum() == (y == 0).sum()          # SMOTE keeps every legit row
    assert list(X_s.columns) == list(X.columns)

    X_u, y_u = resample(X, y, "undersample", ratio=0.1)
    assert y_u.sum() == y.sum()                        # undersampling keeps every fraud row
    assert (y_u == 0).sum() == pytest.approx(y.sum() / 0.1, abs=1)

    X_n, y_n = resample(X, y, "none")
    assert X_n is X and y_n is y


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_train_xgb_each_strategy(data, strategy):
    X_tr, y_tr, X_va, y_va = data
    model = train_xgb(X_tr, y_tr, X_va, y_va, strategy=strategy,
                      params={"n_estimators": 60})
    scores = model.predict_proba(X_va)[:, 1]
    assert scores.shape == (len(X_va),)
    assert model.best_iteration < 60                   # early stopping was active
    expected_spw = scale_pos_weight_value(y_tr) if strategy == "scale_pos_weight" else 1.0
    assert model.get_params()["scale_pos_weight"] == pytest.approx(expected_spw)


def test_random_search_respects_trial_limit(data):
    X_tr, y_tr, X_va, y_va = data
    model, results = random_search(X_tr, y_tr, X_va, y_va, "none", n_trials=3)
    assert len(results) == 3
    assert results["val_pr_auc"].is_monotonic_decreasing
    assert set(train.SEARCH_SPACE) <= set(results.columns)


def test_unknown_strategy_rejected(data):
    X_tr, y_tr, X_va, y_va = data
    with pytest.raises(ValueError):
        train_xgb(X_tr, y_tr, X_va, y_va, strategy="oversample-magic")


def test_compare_strategies_returns_one_row_per_strategy(data):
    X_tr, y_tr, X_va, y_va = data
    results, models, scores = train.compare_strategies(X_tr, y_tr, X_va, y_va,
                                                       strategies=["none", "undersample"])
    assert list(results.index) == ["none", "undersample"]
    assert set(models) == set(scores) == {"none", "undersample"}
    assert results.loc["undersample", "train rows"] < results.loc["none", "train rows"]
    assert results["val PR-AUC"].between(0, 1).all()
