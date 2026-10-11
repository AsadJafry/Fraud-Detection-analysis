"""Tests for src.threshold."""

import numpy as np
import pytest

from src.threshold import best_threshold, cost_curve, cost_of_flagging_nothing

Y = [0, 0, 0, 0, 1, 1]
S = [0.1, 0.2, 0.6, 0.3, 0.7, 0.4]


def test_cost_curve_counts_and_cost():
    curve = cost_curve(Y, S, cost_fn=10, cost_fp=1, thresholds=[0.0, 0.5, 0.65, 0.9])
    # t=0.0 flags all: 4 FP, 0 FN | t=0.5 flags 0.6, 0.7: 1 FP, 1 FN
    # t=0.65 flags 0.7 only: 0 FP, 1 FN | t=0.9 flags nothing: 0 FP, 2 FN
    assert curve["fp"].tolist() == [4, 1, 0, 0]
    assert curve["fn"].tolist() == [0, 1, 1, 2]
    assert curve["total_cost"].tolist() == [4, 11, 10, 20]
    assert curve["precision"].tolist()[:3] == [pytest.approx(2 / 6), 0.5, 1.0]
    assert np.isnan(curve["precision"].iloc[3])      # nothing flagged
    assert curve["recall"].tolist() == [1.0, 0.5, 0.5, 0.0]


def test_threshold_is_inclusive():
    curve = cost_curve([0, 1], [0.2, 0.5], cost_fn=1, cost_fp=1, thresholds=[0.5])
    assert curve.loc[0, "fn"] == 0


def test_best_threshold_minimises_cost():
    t, row = best_threshold(Y, S, cost_fn=10, cost_fp=1, thresholds=[0.0, 0.5, 0.65, 0.9])
    assert t == 0.0 and row["total_cost"] == 4
    # Make false alarms expensive: flagging only the 0.7 score becomes best.
    t, row = best_threshold(Y, S, cost_fn=10, cost_fp=100, thresholds=[0.0, 0.5, 0.65, 0.9])
    assert t == 0.65


def test_best_threshold_picks_middle_of_tied_run():
    y = [0, 1]
    s = [0.1, 0.9]
    t, row = best_threshold(y, s, cost_fn=5, cost_fp=1, thresholds=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    # Every threshold in (0.1, 0.9] is perfect; the tied run is 0.2..0.8 -> middle 0.4 or 0.6.
    assert row["total_cost"] == 0
    assert t in (0.4, 0.6)


def test_only_cost_ratio_matters():
    a, _ = best_threshold(Y, S, cost_fn=500, cost_fp=5)
    b, _ = best_threshold(Y, S, cost_fn=100, cost_fp=1)
    assert a == b


def test_cost_of_flagging_nothing():
    assert cost_of_flagging_nothing(Y, cost_fn=500) == 1000
