"""Tests for MLP estimators."""

from __future__ import annotations

import numpy as np

from src.neural_network import MLPClassifier, MLPRegressor


def test_mlp_classifier_learns_xor() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int_)

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        optimizer="adam",
        learning_rate=0.05,
        max_iter=1_000,
        batch_size=4,
        tol=0.0,
        n_iter_no_change=1_000,
        random_state=42,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(4), atol=1e-6)


def test_mlp_regressor_beats_mean_baseline_on_nonlinear_curve() -> None:
    X = np.linspace(-1.0, 1.0, 30, dtype=np.float64).reshape(-1, 1)
    y = X[:, 0] ** 2
    baseline_error = float(np.mean((y - np.mean(y)) ** 2))

    model = MLPRegressor(
        hidden_layer_sizes=(12,),
        activation="tanh",
        optimizer="adam",
        learning_rate=0.03,
        max_iter=600,
        batch_size=10,
        tol=0.0,
        n_iter_no_change=600,
        random_state=9,
    ).fit(X, y)

    model_error = float(np.mean((model.predict(X) - y) ** 2))
    assert model_error < baseline_error * 0.4


def test_mlp_minibatch_training_is_deterministic_with_seed() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    first = MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        learning_rate=0.02,
        max_iter=20,
        batch_size=2,
        random_state=11,
    ).fit(X, y)
    second = MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        learning_rate=0.02,
        max_iter=20,
        batch_size=2,
        random_state=11,
    ).fit(X, y)

    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


def test_mlp_early_stopping_records_validation_history() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="tanh",
        learning_rate=0.01,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.33,
        n_iter_no_change=2,
        tol=1.0,
        random_state=4,
    ).fit(X, y)

    assert model.converged_
    assert model.n_iter_ < 100
    assert len(model.validation_loss_history_) == model.n_iter_
    assert len(model.learning_rate_history_) == model.n_iter_
