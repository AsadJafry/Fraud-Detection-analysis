"""Tests for src.evaluate."""

import json

import numpy as np
import pytest

from src.evaluate import evaluate_model, save_metrics


def test_threshold_metrics_on_hand_made_scores():
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.6, 0.3, 0.7, 0.4]
    m = evaluate_model(y_true, y_score, threshold=0.5)
    # Predicted fraud: rows 2 (legit) and 4 (fraud).
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["confusion_matrix"] == [[3, 1], [1, 1]]
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["base_rate"] == pytest.approx(2 / 6)


def test_threshold_is_inclusive():
    m = evaluate_model([0, 1], [0.2, 0.5], threshold=0.5)
    assert m["tp"] == 1


def test_perfect_and_constant_scores():
    y = [0, 0, 1, 1]
    perfect = evaluate_model(y, [0.1, 0.2, 0.8, 0.9])
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)

    # A model that scores everything the same ("never fraud") has PR-AUC equal
    # to the base rate, ROC-AUC 0.5, and catches nothing at threshold 0.5.
    constant = evaluate_model(y, [0.0, 0.0, 0.0, 0.0])
    assert constant["pr_auc"] == pytest.approx(0.5)
    assert constant["roc_auc"] == pytest.approx(0.5)
    assert constant["recall"] == 0.0
    assert constant["precision"] == 0.0


def test_metrics_are_json_serialisable(tmp_path):
    m = evaluate_model(np.array([0, 1, 0, 1]), np.array([0.3, 0.8, 0.6, 0.2]))
    path = save_metrics({"model": m}, tmp_path / "m.json")
    assert json.loads(path.read_text())["model"]["n"] == 4
