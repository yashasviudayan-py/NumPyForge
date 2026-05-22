"""Loss functions for neural-network training."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from src.math import clip_probabilities
from src.types import FloatArray


class NeuralLoss(Protocol):
    """Protocol for losses differentiable with respect to predictions."""

    def value(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        """Return scalar loss."""

    def gradient(self, y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
        """Return `d loss / d y_pred`."""


class MeanSquaredErrorLoss:
    """Mean squared error loss."""

    def value(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def gradient(self, y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
        return cast(FloatArray, 2.0 * (y_pred - y_true) / y_true.size)


class BinaryCrossEntropyLoss:
    """Binary cross-entropy for sigmoid probabilities."""

    def value(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        clipped = clip_probabilities(y_pred)
        losses = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
        return float(np.mean(losses))

    def gradient(self, y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
        clipped = clip_probabilities(y_pred)
        n_outputs = y_true.size
        return cast(FloatArray, (clipped - y_true) / (clipped * (1.0 - clipped) * n_outputs))


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy for softmax probabilities."""

    def value(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        clipped = clip_probabilities(y_pred)
        return float(-np.mean(np.sum(y_true * np.log(clipped), axis=1)))

    def gradient(self, y_true: FloatArray, y_pred: FloatArray) -> FloatArray:
        clipped = clip_probabilities(y_pred)
        return cast(FloatArray, -y_true / (clipped * y_true.shape[0]))
