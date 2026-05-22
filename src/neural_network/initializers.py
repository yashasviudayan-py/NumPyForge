"""Parameter initializers for pure NumPy neural networks."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from src.types import FloatArray

Initializer = Literal["zeros", "normal", "xavier", "he"]


def initialize_parameters(
    shape: tuple[int, ...],
    *,
    initializer: Initializer,
    rng: np.random.Generator,
) -> FloatArray:
    """Return an initialized float64 parameter array."""
    if len(shape) == 0:
        raise ValueError("shape must contain at least one dimension.")

    if initializer == "zeros":
        return np.zeros(shape, dtype=np.float64)

    if initializer == "normal":
        return cast(FloatArray, rng.normal(loc=0.0, scale=0.01, size=shape).astype(np.float64))

    fan_in, fan_out = _fan_in_out(shape)
    if initializer == "xavier":
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return cast(FloatArray, rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float64))

    if initializer == "he":
        scale = np.sqrt(2.0 / fan_in)
        return cast(FloatArray, rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float64))

    raise ValueError("initializer must be 'zeros', 'normal', 'xavier', or 'he'.")


def _fan_in_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 1:
        return shape[0], shape[0]

    return shape[0], shape[1]
