"""Finite-difference checks for neural-network backpropagation."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.neural_network.gradient_check import gradient_check
from src.neural_network.layers import Dense, TanhActivation
from src.neural_network.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    MeanSquaredErrorLoss,
)
from src.neural_network.network import SequentialNetwork
from src.types import FloatArray


def test_dense_weight_gradient_passes_finite_difference_check() -> None:
    X = np.array([[0.2, -0.4]], dtype=np.float64)
    grad_output = np.array([[0.7]], dtype=np.float64)
    layer = Dense(2, 1, rng=np.random.default_rng(3))

    def value_fn(weights: FloatArray) -> float:
        layer.weights[:] = weights.reshape(2, 1)
        return float(np.sum(layer.forward(X, training=True) * grad_output))

    def grad_fn(weights: FloatArray) -> FloatArray:
        layer.weights[:] = weights.reshape(2, 1)
        layer.forward(X, training=True)
        layer.backward(grad_output)
        return cast(FloatArray, layer.grads()["weights"].reshape(weights.shape))

    assert gradient_check(value_fn, grad_fn, layer.weights.copy())


def test_loss_gradients_pass_finite_difference_checks() -> None:
    mse = MeanSquaredErrorLoss()
    bce = BinaryCrossEntropyLoss()
    cce = CategoricalCrossEntropyLoss()

    y_reg = np.array([[0.2, -0.1]], dtype=np.float64)
    y_binary = np.array([[1.0], [0.0]], dtype=np.float64)
    y_class = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    assert gradient_check(
        lambda pred: mse.value(y_reg, pred.reshape(y_reg.shape)),
        lambda pred: cast(
            FloatArray, mse.gradient(y_reg, pred.reshape(y_reg.shape)).reshape(pred.shape)
        ),
        np.array([[0.3, -0.2]], dtype=np.float64),
    )
    assert gradient_check(
        lambda pred: bce.value(y_binary, pred.reshape(y_binary.shape)),
        lambda pred: cast(
            FloatArray, bce.gradient(y_binary, pred.reshape(y_binary.shape)).reshape(pred.shape)
        ),
        np.array([[0.8], [0.2]], dtype=np.float64),
    )
    assert gradient_check(
        lambda pred: cce.value(y_class, pred.reshape(y_class.shape)),
        lambda pred: cast(
            FloatArray, cce.gradient(y_class, pred.reshape(y_class.shape)).reshape(pred.shape)
        ),
        np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64),
    )


def test_tiny_mlp_backprop_passes_finite_difference_check() -> None:
    X = np.array([[0.5, -0.25]], dtype=np.float64)
    y = np.array([[0.1]], dtype=np.float64)
    first = Dense(2, 3, rng=np.random.default_rng(5))
    second = Dense(3, 1, rng=np.random.default_rng(6))
    network = SequentialNetwork([first, TanhActivation(), second])
    loss = MeanSquaredErrorLoss()

    def value_fn(weights: FloatArray) -> float:
        first.weights[:] = weights.reshape(first.weights.shape)
        predictions = network.forward(X, training=True)
        return loss.value(y, predictions)

    def grad_fn(weights: FloatArray) -> FloatArray:
        first.weights[:] = weights.reshape(first.weights.shape)
        predictions = network.forward(X, training=True)
        network.backward(loss.gradient(y, predictions))
        return cast(FloatArray, first.grads()["weights"].reshape(weights.shape))

    assert gradient_check(value_fn, grad_fn, first.weights.copy(), tolerance=1e-3)
