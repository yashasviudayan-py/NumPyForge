"""Neural-network layers and activations implemented with NumPy."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from src.math import softmax, stable_sigmoid
from src.neural_network.initializers import Initializer, initialize_parameters
from src.types import FloatArray


class Layer(Protocol):
    """Protocol for layers participating in forward/backward propagation."""

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        """Return layer output for input `X`."""

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Backpropagate output gradients and return input gradients."""

    def params(self) -> dict[str, FloatArray]:
        """Return trainable parameters."""

    def grads(self) -> dict[str, FloatArray]:
        """Return gradients matching trainable parameters."""


class Dense:
    """Fully connected layer: `Y = X @ W + b`."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        initializer: Initializer = "xavier",
        rng: np.random.Generator,
    ) -> None:
        if n_inputs <= 0 or n_outputs <= 0:
            raise ValueError("n_inputs and n_outputs must be positive.")

        self.weights = initialize_parameters(
            (n_inputs, n_outputs), initializer=initializer, rng=rng
        )
        self.bias = np.zeros(n_outputs, dtype=np.float64)
        self._input: FloatArray | None = None
        self._grad_weights = np.zeros_like(self.weights, dtype=np.float64)
        self._grad_bias = np.zeros_like(self.bias, dtype=np.float64)

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._input = X
        return cast(FloatArray, X @ self.weights + self.bias)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._input is None:
            raise RuntimeError("Dense.backward called before forward.")

        self._grad_weights = cast(FloatArray, self._input.T @ grad_output)
        self._grad_bias = cast(FloatArray, np.sum(grad_output, axis=0))
        return cast(FloatArray, grad_output @ self.weights.T)

    def params(self) -> dict[str, FloatArray]:
        return {"weights": self.weights, "bias": self.bias}

    def grads(self) -> dict[str, FloatArray]:
        return {"weights": self._grad_weights, "bias": self._grad_bias}


class ReLUActivation:
    """Rectified linear unit activation."""

    def __init__(self) -> None:
        self._input: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._input = X
        return cast(FloatArray, np.maximum(X, 0.0))

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._input is None:
            raise RuntimeError("ReLUActivation.backward called before forward.")
        return cast(FloatArray, grad_output * (self._input > 0.0))

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}


class LeakyReLUActivation:
    """Leaky ReLU activation."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        if negative_slope < 0.0:
            raise ValueError("negative_slope cannot be negative.")
        self.negative_slope = negative_slope
        self._input: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._input = X
        return cast(FloatArray, np.where(X > 0.0, X, self.negative_slope * X))

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._input is None:
            raise RuntimeError("LeakyReLUActivation.backward called before forward.")
        local_gradient = np.where(self._input > 0.0, 1.0, self.negative_slope)
        return cast(FloatArray, grad_output * local_gradient)

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}


class SigmoidActivation:
    """Sigmoid activation."""

    def __init__(self) -> None:
        self._output: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._output = stable_sigmoid(X)
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._output is None:
            raise RuntimeError("SigmoidActivation.backward called before forward.")
        return cast(FloatArray, grad_output * self._output * (1.0 - self._output))

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}


class TanhActivation:
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self._output: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._output = cast(FloatArray, np.tanh(X))
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._output is None:
            raise RuntimeError("TanhActivation.backward called before forward.")
        return cast(FloatArray, grad_output * (1.0 - self._output * self._output))

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}


class SoftmaxActivation:
    """Softmax activation over the last axis."""

    def __init__(self) -> None:
        self._output: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        del training
        self._output = softmax(X, axis=1)
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._output is None:
            raise RuntimeError("SoftmaxActivation.backward called before forward.")

        dot = np.sum(grad_output * self._output, axis=1, keepdims=True)
        return cast(FloatArray, self._output * (grad_output - dot))

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}


class Dropout:
    """Inverted dropout layer used only during training."""

    def __init__(self, dropout_rate: float, *, rng: np.random.Generator) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0).")
        self.dropout_rate = dropout_rate
        self.rng = rng
        self._mask: FloatArray | None = None

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        if not training or self.dropout_rate == 0.0:
            self._mask = np.ones_like(X, dtype=np.float64)
            return X

        keep_probability = 1.0 - self.dropout_rate
        mask = self.rng.binomial(1, keep_probability, size=X.shape).astype(np.float64)
        self._mask = cast(FloatArray, mask / keep_probability)
        return cast(FloatArray, X * self._mask)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self._mask is None:
            raise RuntimeError("Dropout.backward called before forward.")
        return cast(FloatArray, grad_output * self._mask)

    def params(self) -> dict[str, FloatArray]:
        return {}

    def grads(self) -> dict[str, FloatArray]:
        return {}
