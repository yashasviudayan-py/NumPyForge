"""Deterministic random-state helpers."""

from __future__ import annotations

import numpy as np

from src.types import RandomState


def check_random_state(random_state: RandomState = None) -> np.random.Generator:
    """Return a NumPy random generator from a seed, generator, or `None`.

    Args:
        random_state: Integer seed for deterministic behavior, an existing
            `np.random.Generator`, or `None` for NumPy's entropy source.

    Returns:
        A NumPy `Generator` suitable for model initialization and data shuffling.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state

    if random_state is None or isinstance(random_state, int):
        return np.random.default_rng(random_state)

    raise TypeError("random_state must be an int, numpy.random.Generator, or None.")
