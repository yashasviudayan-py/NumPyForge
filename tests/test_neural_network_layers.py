"""Tests for neural-network layers, losses, and initializers."""

from __future__ import annotations

import numpy as np

from src.neural_network.initializers import initialize_parameters
from src.neural_network.layers import (
    Dense,
    Dropout,
    Layer,
    LeakyReLUActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
    TanhActivation,
)
from src.neural_network.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    MeanSquaredErrorLoss,
)


def test_dense_forward_backward_shapes() -> None:
    rng = np.random.default_rng(42)
    layer = Dense(3, 2, rng=rng)
    X = np.ones((4, 3), dtype=np.float64)
    output = layer.forward(X, training=True)
    grad_input = layer.backward(np.ones_like(output))

    assert output.shape == (4, 2)
    assert grad_input.shape == X.shape
    assert layer.grads()["weights"].shape == (3, 2)
    assert layer.grads()["bias"].shape == (2,)


def test_activation_layers_forward_backward_shapes() -> None:
    X = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)
    grad = np.ones_like(X)

    activations: list[Layer] = [
        ReLUActivation(),
        LeakyReLUActivation(),
        SigmoidActivation(),
        TanhActivation(),
        SoftmaxActivation(),
    ]
    for activation in activations:
        output = activation.forward(X, training=True)
        grad_input = activation.backward(grad)
        assert output.shape == X.shape
        assert grad_input.shape == X.shape


def test_dropout_masks_only_during_training() -> None:
    X = np.ones((100, 4), dtype=np.float64)
    dropout = Dropout(0.5, rng=np.random.default_rng(1))

    train_output = dropout.forward(X, training=True)
    eval_output = dropout.forward(X, training=False)

    assert np.any(train_output == 0.0)
    np.testing.assert_array_equal(eval_output, X)


def test_initializers_are_deterministic_with_fixed_seed() -> None:
    first = initialize_parameters((3, 2), initializer="xavier", rng=np.random.default_rng(7))
    second = initialize_parameters((3, 2), initializer="xavier", rng=np.random.default_rng(7))

    np.testing.assert_allclose(first, second)
    assert initialize_parameters(
        (3, 2), initializer="zeros", rng=np.random.default_rng(7)
    ).shape == (
        3,
        2,
    )


def test_loss_gradients_match_prediction_shapes() -> None:
    y_binary = np.array([[0.0], [1.0]], dtype=np.float64)
    p_binary = np.array([[0.25], [0.75]], dtype=np.float64)
    y_multiclass = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    p_multiclass = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)

    assert MeanSquaredErrorLoss().gradient(y_binary, p_binary).shape == p_binary.shape
    assert BinaryCrossEntropyLoss().gradient(y_binary, p_binary).shape == p_binary.shape
    assert CategoricalCrossEntropyLoss().gradient(y_multiclass, p_multiclass).shape == (
        2,
        2,
    )
