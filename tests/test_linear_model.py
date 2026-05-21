"""Tests for custom linear models."""

from __future__ import annotations

import numpy as np
import pytest

from src.linear_model import LinearRegression, LogisticRegression


def test_linear_regression_normal_equation_recovers_known_coefficients() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]], dtype=np.float64)
    y = 1.5 + X @ np.array([2.0, -0.5], dtype=np.float64)

    model = LinearRegression().fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)
    assert model.bias_ == pytest.approx(1.5)


def test_linear_regression_l2_shrinks_coefficients() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)

    unregularized = LinearRegression(penalty=None).fit(X, y)
    ridge = LinearRegression(penalty="l2", regularization_strength=20.0).fit(X, y)

    assert ridge.weights_ is not None
    assert unregularized.weights_ is not None
    assert abs(ridge.weights_[0]) < abs(unregularized.weights_[0])


def test_linear_regression_gradient_descent_converges_near_closed_form() -> None:
    X = np.linspace(-2.0, 2.0, 25, dtype=np.float64).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 1.0
    closed_form = LinearRegression().fit(X, y)

    gradient_model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.1,
        n_iterations=2_000,
        tol=0.0,
        gradient_tol=1e-8,
    ).fit(X, y)

    assert gradient_model.converged_
    np.testing.assert_allclose(gradient_model.predict(X), closed_form.predict(X), atol=1e-2)


def test_linear_regression_l1_encourages_smaller_coefficients() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    unregularized = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.05,
        n_iterations=800,
        penalty=None,
    ).fit(X, y)
    l1_model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.05,
        n_iterations=800,
        penalty="l1",
        regularization_strength=4.0,
    ).fit(X, y)

    assert l1_model.weights_ is not None
    assert unregularized.weights_ is not None
    assert abs(l1_model.weights_[0]) < abs(unregularized.weights_[0])


def test_linear_regression_rejects_l1_with_normal_equation() -> None:
    with pytest.raises(ValueError, match="only supported"):
        LinearRegression(solver="normal_equation", penalty="l1")


def test_linear_regression_sample_weights_affect_fit() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 10.0], dtype=np.float64)

    unweighted = LinearRegression().fit(X, y)
    weighted = LinearRegression().fit(X, y, sample_weight=np.array([10.0, 10.0, 1.0]))

    assert abs(weighted.predict([[1.0]])[0] - 1.0) < abs(unweighted.predict([[1.0]])[0] - 1.0)


def test_logistic_regression_fit_predicts_binary_labels() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int_)

    model = LogisticRegression(learning_rate=0.1, n_iterations=2_000, random_state=42)
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert model.is_fitted
    assert probabilities.shape == (4, 2)
    assert predictions.shape == y.shape
    assert set(int(prediction) for prediction in predictions).issubset({0, 1})


def test_logistic_regression_multiclass_softmax_fits_synthetic_clusters() -> None:
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -2.0],
            [2.0, -2.0],
            [2.5, -1.5],
            [0.0, 2.0],
            [0.5, 2.5],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int_)

    model = LogisticRegression(
        learning_rate=0.5,
        n_iterations=1_000,
        multi_class="multinomial",
        random_state=7,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    assert model.predict_proba(X).shape == (6, 3)


def test_logistic_regression_l2_and_l1_reduce_parameter_norms() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    base = LogisticRegression(
        learning_rate=0.2,
        n_iterations=500,
        penalty=None,
        random_state=3,
    ).fit(X, y)
    l2_model = LogisticRegression(
        learning_rate=0.2,
        n_iterations=500,
        penalty="l2",
        regularization_strength=20.0,
        random_state=3,
    ).fit(X, y)
    l1_model = LogisticRegression(
        learning_rate=0.2,
        n_iterations=500,
        penalty="l1",
        regularization_strength=20.0,
        random_state=3,
    ).fit(X, y)

    assert base.weights_ is not None
    assert l2_model.weights_ is not None
    assert l1_model.weights_ is not None
    assert np.linalg.norm(l2_model.weights_) < np.linalg.norm(base.weights_)
    assert np.sum(np.abs(l1_model.weights_)) < np.sum(np.abs(base.weights_))


def test_logistic_regression_class_weights_and_sample_weights_affect_fit() -> None:
    X = np.array([[0.0], [0.2], [0.4], [0.6], [0.8], [3.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 0, 1], dtype=np.int_)

    unweighted = LogisticRegression(learning_rate=0.2, n_iterations=400, random_state=9).fit(X, y)
    weighted = LogisticRegression(
        learning_rate=0.2,
        n_iterations=400,
        class_weight={0: 1.0, 1: 20.0},
        random_state=9,
    ).fit(X, y, sample_weight=np.ones_like(y, dtype=np.float64))

    assert weighted.predict_proba([[2.0]])[0, 1] > unweighted.predict_proba([[2.0]])[0, 1]


def test_logistic_regression_minibatch_is_deterministic_with_fixed_seed() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    first = LogisticRegression(
        learning_rate=0.1,
        n_iterations=100,
        batch_strategy="mini_batch",
        batch_size=2,
        random_state=11,
    ).fit(X, y)
    second = LogisticRegression(
        learning_rate=0.1,
        n_iterations=100,
        batch_strategy="mini_batch",
        batch_size=2,
        random_state=11,
    ).fit(X, y)

    assert first.weights_ is not None
    assert second.weights_ is not None
    assert first.bias_ is not None
    assert second.bias_ is not None
    np.testing.assert_allclose(first.weights_, second.weights_)
    np.testing.assert_allclose(first.bias_, second.bias_)


def test_logistic_regression_early_stopping_records_histories() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    model = LogisticRegression(
        learning_rate=0.1,
        n_iterations=200,
        early_stopping=True,
        validation_fraction=0.33,
        n_iter_no_change=3,
        tol=1.0,
        random_state=5,
    ).fit(X, y)

    assert model.converged_
    assert model.n_iter_ < 200
    assert len(model.loss_history_) == model.n_iter_
    assert len(model.validation_loss_history_) == model.n_iter_
    assert len(model.gradient_norm_history_) == model.n_iter_
