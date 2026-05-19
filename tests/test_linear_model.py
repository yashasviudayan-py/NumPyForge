"""Tests for custom linear models."""

from __future__ import annotations

import numpy as np

from src.linear_model import LogisticRegression


def test_logistic_regression_fit_predicts_binary_labels() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int_)

    model = LogisticRegression(learning_rate=0.1, n_iterations=2_000, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)

    assert model.is_fitted
    assert predictions.shape == y.shape
    assert set(int(prediction) for prediction in predictions).issubset({0, 1})
