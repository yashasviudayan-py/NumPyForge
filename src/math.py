"""Vectorized numerical utilities used across NumPyForge.

These helpers keep numerically sensitive operations in one place. They favor
stable formulations such as clipped logits and log-sum-exp normalization, which
will matter more as later phases add multiclass models and neural networks.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from src.types import FloatArray, IntArray


def stable_sigmoid(values: FloatArray) -> FloatArray:
    """Compute the sigmoid function with clipping to avoid overflow.

    Args:
        values: Input array of logits with any shape.

    Returns:
        Array with the same shape containing probabilities in `(0, 1)`.
    """
    clipped = np.clip(values, -500.0, 500.0)
    return cast(FloatArray, 1.0 / (1.0 + np.exp(-clipped)))


def log_sum_exp(
    values: FloatArray, *, axis: int | None = None, keepdims: bool = False
) -> FloatArray:
    """Compute `log(sum(exp(values)))` using a stable max-shift formulation."""
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = max_values + np.log(summed)

    if keepdims:
        return cast(FloatArray, result)

    return cast(FloatArray, np.squeeze(result, axis=axis))


def softmax(values: FloatArray, *, axis: int = -1) -> FloatArray:
    """Compute row-wise or axis-wise softmax probabilities."""
    normalized = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(normalized)
    return cast(FloatArray, exp_values / np.sum(exp_values, axis=axis, keepdims=True))


def one_hot(labels: IntArray, *, n_classes: int | None = None) -> FloatArray:
    """Return one-hot encoded labels with shape `(n_samples, n_classes)`."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array; got shape {labels.shape}.")

    if labels.shape[0] == 0:
        raise ValueError("labels must contain at least one value.")

    if np.any(labels < 0):
        raise ValueError("labels must be non-negative integers.")

    inferred_classes = int(np.max(labels)) + 1
    class_count = inferred_classes if n_classes is None else n_classes

    if class_count < inferred_classes:
        raise ValueError("n_classes must be greater than the largest label.")

    encoded = np.zeros((labels.shape[0], class_count), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def clip_probabilities(probabilities: FloatArray, *, eps: float = 1e-15) -> FloatArray:
    """Clip probabilities into `[eps, 1 - eps]` for stable log-loss terms."""
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5.")

    return cast(FloatArray, np.clip(probabilities, eps, 1.0 - eps))


def l2_norm(values: FloatArray) -> float:
    """Return the Euclidean norm of a vector or matrix as a Python float."""
    return float(np.linalg.norm(values))


def squared_l2_norm(values: FloatArray) -> float:
    """Return the squared Euclidean norm of a vector or matrix."""
    return float(np.sum(values * values))
