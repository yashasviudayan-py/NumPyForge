"""Finite-difference checks for classical model gradients."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.linear_model import LogisticRegression
from src.neural_network.gradient_check import gradient_check
from src.optimizers import OptimizerParameters
from src.types import FloatArray


def test_binary_logistic_regression_gradient_matches_finite_difference() -> None:
    X = np.array([[0.2, -0.5], [1.0, 0.3], [-0.7, 0.8]], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    model = LogisticRegression(penalty=None, random_state=3)

    def unpack(flat_parameters: FloatArray) -> OptimizerParameters:
        return {
            "weights": cast(FloatArray, flat_parameters[:2].copy()),
            "bias": cast(FloatArray, flat_parameters[2:].copy()),
        }

    def value_fn(flat_parameters: FloatArray) -> float:
        return model._binary_loss(unpack(flat_parameters), X, y, sample_weight=None)

    def grad_fn(flat_parameters: FloatArray) -> FloatArray:
        gradient = model._binary_gradient(unpack(flat_parameters), X, y, sample_weight=None)
        return cast(FloatArray, np.concatenate((gradient["weights"], gradient["bias"])))

    initial = np.array([0.1, -0.2, 0.05], dtype=np.float64)

    assert gradient_check(value_fn, grad_fn, initial)
