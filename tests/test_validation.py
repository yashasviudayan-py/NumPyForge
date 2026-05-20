"""Tests for shared validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation import (
    check_binary_targets,
    check_class_labels,
    check_feature_matrix,
    check_matching_n_features,
    check_sample_weight,
    check_target_vector,
)


def test_check_feature_matrix_coerces_2d_float_array() -> None:
    features = check_feature_matrix([[1, 2], [3, 4]])

    assert features.dtype == np.float64
    assert features.shape == (2, 2)


def test_check_feature_matrix_rejects_bad_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="2D array"):
        check_feature_matrix([1, 2, 3])

    with pytest.raises(ValueError, match="finite"):
        check_feature_matrix([[1.0, np.nan]])


def test_check_target_vector_validates_length() -> None:
    targets = check_target_vector(np.array([1, 0], dtype=np.int_), n_samples=2)

    np.testing.assert_array_equal(targets, np.array([1, 0], dtype=np.int_))

    with pytest.raises(ValueError, match="inconsistent sample counts"):
        check_target_vector(np.array([1], dtype=np.int_), n_samples=2)


def test_check_binary_targets_accepts_only_zero_one_encoding() -> None:
    targets = check_binary_targets(np.array([0, 1, 1], dtype=np.int_))

    np.testing.assert_array_equal(targets, np.array([0.0, 1.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match="0 and 1"):
        check_binary_targets(np.array([0, 2], dtype=np.int_))


def test_check_class_labels_requires_integer_dtype() -> None:
    labels = check_class_labels(np.array([0, 2, 1], dtype=np.int_))

    np.testing.assert_array_equal(labels, np.array([0, 2, 1], dtype=np.int_))

    with pytest.raises(ValueError, match="integer dtype"):
        check_class_labels(np.array([0.0, 1.0], dtype=np.float64))


def test_check_sample_weight_validates_shape_and_values() -> None:
    weights = check_sample_weight([1.0, 0.5], n_samples=2)

    assert weights is not None
    np.testing.assert_array_equal(weights, np.array([1.0, 0.5], dtype=np.float64))

    with pytest.raises(ValueError, match="cannot contain negative"):
        check_sample_weight([1.0, -1.0], n_samples=2)


def test_check_matching_n_features_rejects_prediction_width_mismatch() -> None:
    features = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="fitted with 3 features"):
        check_matching_n_features(features, n_features_in=3)
