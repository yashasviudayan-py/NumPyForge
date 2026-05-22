"""Tests for reusable optimization routines."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.optimizers import (
    AdamOptimizer,
    GradientDescentConfig,
    LearningRateSchedule,
    MomentumOptimizer,
    OptimizerParameters,
    RMSPropOptimizer,
    SGDOptimizer,
    run_gradient_descent,
)
from src.types import ArrayLike, FloatArray


def quadratic_objective(
    parameters: OptimizerParameters,
    X: FloatArray,
    y: ArrayLike,
    sample_weight: FloatArray | None,
) -> float:
    del X, y, sample_weight
    value = parameters["x"] - 3.0
    return float(np.sum(value * value))


def quadratic_gradient(
    parameters: OptimizerParameters,
    X: FloatArray,
    y: ArrayLike,
    sample_weight: FloatArray | None,
) -> OptimizerParameters:
    del X, y, sample_weight
    return {"x": cast(FloatArray, 2.0 * (parameters["x"] - 3.0))}


def test_gradient_descent_decreases_quadratic_loss() -> None:
    X = np.ones((4, 1), dtype=np.float64)
    y = np.ones(4, dtype=np.float64)

    result = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(learning_rate=0.1, max_iter=20, tol=0.0),
    )

    assert result.loss_history[-1] < result.loss_history[0]
    assert result.parameters["x"][0] > 2.9


def test_gradient_norm_stopping_triggers_convergence() -> None:
    X = np.ones((2, 1), dtype=np.float64)
    y = np.ones(2, dtype=np.float64)

    result = run_gradient_descent(
        initial_parameters={"x": np.array([3.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(gradient_tol=1e-12),
    )

    assert result.converged
    assert result.n_iter == 1


def test_no_improvement_stopping_triggers_convergence() -> None:
    X = np.ones((2, 1), dtype=np.float64)
    y = np.ones(2, dtype=np.float64)

    result = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(
            learning_rate=0.0 + 1e-12,
            max_iter=100,
            tol=1.0,
            n_iter_no_change=2,
        ),
    )

    assert result.converged
    assert result.n_iter == 3


def test_minibatch_and_stochastic_modes_are_deterministic_with_fixed_seed() -> None:
    X = np.ones((5, 1), dtype=np.float64)
    y = np.ones(5, dtype=np.float64)

    first = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(
            learning_rate=0.05,
            max_iter=5,
            batch_strategy="mini_batch",
            batch_size=2,
            random_state=42,
        ),
    )
    second = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(
            learning_rate=0.05,
            max_iter=5,
            batch_strategy="mini_batch",
            batch_size=2,
            random_state=42,
        ),
    )
    stochastic = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=GradientDescentConfig(
            learning_rate=0.05,
            max_iter=5,
            batch_strategy="stochastic",
            random_state=42,
        ),
    )

    np.testing.assert_allclose(first.parameters["x"], second.parameters["x"])
    assert stochastic.loss_history[-1] < stochastic.loss_history[0]


def test_validation_split_is_deterministic() -> None:
    X = np.ones((8, 1), dtype=np.float64)
    y = np.ones(8, dtype=np.float64)
    config = GradientDescentConfig(
        learning_rate=0.05,
        max_iter=5,
        early_stopping=True,
        validation_fraction=0.25,
        random_state=13,
    )

    first = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=config,
    )
    second = run_gradient_descent(
        initial_parameters={"x": np.array([0.0], dtype=np.float64)},
        X=X,
        y=y,
        objective=quadratic_objective,
        gradient=quadratic_gradient,
        config=config,
    )

    assert first.validation_loss_history == second.validation_loss_history


def test_stateful_optimizers_match_hand_computed_first_steps() -> None:
    parameters = {"w": np.array([1.0, -1.0], dtype=np.float64)}
    gradients = {"w": np.array([0.5, -0.25], dtype=np.float64)}

    sgd_parameters = {"w": parameters["w"].copy()}
    SGDOptimizer().step(sgd_parameters, gradients, learning_rate=0.1)
    np.testing.assert_allclose(sgd_parameters["w"], np.array([0.95, -0.975]))

    momentum_parameters = {"w": parameters["w"].copy()}
    MomentumOptimizer(momentum=0.9).step(momentum_parameters, gradients, learning_rate=0.1)
    np.testing.assert_allclose(momentum_parameters["w"], np.array([0.95, -0.975]))

    rmsprop_parameters = {"w": parameters["w"].copy()}
    RMSPropOptimizer(decay_rate=0.0, eps=0.0 + 1e-8).step(
        rmsprop_parameters, gradients, learning_rate=0.1
    )
    np.testing.assert_allclose(rmsprop_parameters["w"], np.array([0.9, -0.9]), atol=1e-7)

    adam_parameters = {"w": parameters["w"].copy()}
    AdamOptimizer(beta1=0.0, beta2=0.0, eps=0.0 + 1e-8).step(
        adam_parameters, gradients, learning_rate=0.1
    )
    np.testing.assert_allclose(adam_parameters["w"], np.array([0.9, -0.9]), atol=1e-7)


def test_learning_rate_schedules_return_expected_values() -> None:
    constant = LearningRateSchedule(initial_learning_rate=0.1, kind="constant")
    step = LearningRateSchedule(
        initial_learning_rate=0.1,
        kind="step",
        step_size=2,
        decay_rate=0.5,
    )
    cosine = LearningRateSchedule(initial_learning_rate=0.1, kind="cosine", max_iter=10)

    assert constant.value(7) == pytest.approx(0.1)
    assert step.value(0) == pytest.approx(0.1)
    assert step.value(2) == pytest.approx(0.05)
    assert cosine.value(0) == pytest.approx(0.1)
    assert cosine.value(10) == pytest.approx(0.0)
