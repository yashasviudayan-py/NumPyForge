"""Tests for vectorized numerical helpers."""

from __future__ import annotations

import numpy as np

from src.math import (
    clip_probabilities,
    l2_norm,
    log_sum_exp,
    one_hot,
    softmax,
    squared_l2_norm,
    stable_sigmoid,
)


def test_stable_sigmoid_handles_extreme_logits() -> None:
    logits = np.array([-1_000.0, 0.0, 1_000.0], dtype=np.float64)

    probabilities = stable_sigmoid(logits)

    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] < 1e-200
    assert probabilities[1] == 0.5
    assert probabilities[2] > 1.0 - 1e-12


def test_softmax_normalizes_each_row() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1_000.0, 1_001.0, 1_002.0]], dtype=np.float64)

    probabilities = softmax(logits, axis=1)

    np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))
    assert np.all(probabilities > 0.0)


def test_log_sum_exp_matches_direct_computation_on_safe_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    result = log_sum_exp(values, axis=1)
    expected = np.log(np.sum(np.exp(values), axis=1))

    np.testing.assert_allclose(result, expected)


def test_one_hot_encodes_integer_labels() -> None:
    labels = np.array([2, 0, 1], dtype=np.int_)

    encoded = one_hot(labels, n_classes=3)

    np.testing.assert_array_equal(
        encoded,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_clip_probabilities_and_norm_helpers() -> None:
    probabilities = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    values = np.array([3.0, 4.0], dtype=np.float64)

    clipped = clip_probabilities(probabilities, eps=1e-6)

    np.testing.assert_allclose(clipped, np.array([1e-6, 0.5, 1.0 - 1e-6]))
    assert l2_norm(values) == 5.0
    assert squared_l2_norm(values) == 25.0
