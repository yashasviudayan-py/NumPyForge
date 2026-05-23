"""Tests for evaluation metrics and reports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.metrics import (
    accuracy_score,
    adjusted_r2_score,
    classification_report_dict,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    pr_auc_score,
    precision_score,
    r2_score,
    recall_score,
    regression_report_dict,
    roc_auc_score,
    root_mean_squared_error,
)


def test_classification_metrics_match_hand_computed_values() -> None:
    y_true = np.array([0, 1, 1, 0], dtype=np.int_)
    y_pred = np.array([0, 1, 0, 0], dtype=np.int_)

    assert accuracy_score(y_true, y_pred) == 0.75
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), np.array([[2, 0], [1, 1]]))
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 0.5
    assert f1_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_multiclass_average_modes_handle_zero_division() -> None:
    y_true = np.array([0, 1, 2, 2], dtype=np.int_)
    y_pred = np.array([0, 0, 0, 2], dtype=np.int_)

    assert precision_score(y_true, y_pred, average="macro") == pytest.approx((1 / 3 + 0 + 1) / 3)
    assert recall_score(y_true, y_pred, average="micro") == pytest.approx(0.5)
    assert f1_score(y_true, y_pred, average="weighted") >= 0.0


def test_probability_metrics_match_known_examples() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int_)
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    y_proba = np.column_stack((1.0 - y_score, y_score))

    assert roc_auc_score(y_true, y_score) == pytest.approx(0.75)
    assert pr_auc_score(y_true, y_score) > 0.0
    assert log_loss(y_true, y_proba) > 0.0


def test_regression_metrics_match_hand_computed_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y_pred = np.array([1.0, 2.0, 4.0], dtype=np.float64)

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0 / 3.0)
    assert mean_squared_error(y_true, y_pred) == pytest.approx(1.0 / 3.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(1.0 / 3.0))
    assert r2_score(y_true, y_pred) == pytest.approx(0.5)
    assert adjusted_r2_score(y_true, y_pred, n_features=1) == pytest.approx(0.0)
    assert explained_variance_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_reports_are_json_serializable() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int_)
    y_pred = np.array([0, 0, 1, 0], dtype=np.int_)
    y_score = np.array([0.1, 0.2, 0.8, 0.4], dtype=np.float64)

    classification = classification_report_dict(y_true, y_pred, y_score)
    regression = regression_report_dict(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 4.0]),
        n_features=1,
    )

    json.dumps(classification, allow_nan=False)
    json.dumps(regression, allow_nan=False)
