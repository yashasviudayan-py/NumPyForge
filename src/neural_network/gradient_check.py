"""Finite-difference gradient checking utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.types import FloatArray


def gradient_check(
    value_fn: Callable[[FloatArray], float],
    grad_fn: Callable[[FloatArray], FloatArray],
    parameters: FloatArray,
    *,
    eps: float = 1e-5,
    tolerance: float = 1e-4,
) -> bool:
    """Compare analytic gradients against central finite differences."""
    numerical = np.zeros_like(parameters, dtype=np.float64)

    for index in np.ndindex(parameters.shape):
        original = float(parameters[index])

        parameters[index] = original + eps
        plus = value_fn(parameters)
        parameters[index] = original - eps
        minus = value_fn(parameters)
        parameters[index] = original

        numerical[index] = (plus - minus) / (2.0 * eps)

    analytic = grad_fn(parameters)
    return bool(np.allclose(numerical, analytic, atol=tolerance, rtol=tolerance))
