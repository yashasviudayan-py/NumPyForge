"""Internal sequential network runner."""

from __future__ import annotations

from src.neural_network.layers import Layer
from src.optimizers import OptimizerParameters
from src.types import FloatArray


class SequentialNetwork:
    """Simple ordered layer container with parameter collection helpers."""

    def __init__(self, layers: list[Layer]) -> None:
        if not layers:
            raise ValueError("SequentialNetwork requires at least one layer.")
        self.layers = layers

    def forward(self, X: FloatArray, *, training: bool) -> FloatArray:
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        gradient = grad_output
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> OptimizerParameters:
        parameters: OptimizerParameters = {}
        for index, layer in enumerate(self.layers):
            for name, value in layer.params().items():
                parameters[f"{index}.{name}"] = value
        return parameters

    def grads(self) -> OptimizerParameters:
        gradients: OptimizerParameters = {}
        for index, layer in enumerate(self.layers):
            for name, value in layer.grads().items():
                gradients[f"{index}.{name}"] = value
        return gradients
