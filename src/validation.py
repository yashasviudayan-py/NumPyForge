"""Reusable validation helpers for estimator inputs.

The framework keeps validation outside individual estimators so models share
consistent error messages, shape handling, and finite-value guarantees.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from src.types import ArrayLike, FloatArray, IntArray, RawArrayLike


def check_feature_matrix(X: RawArrayLike, *, ensure_finite: bool = True) -> FloatArray:
    """Validate and coerce features to a 2D float64 matrix.

    Args:
        X: Feature matrix with shape `(n_samples, n_features)`.
        ensure_finite: Whether to reject NaN and infinite values.

    Returns:
        A float64 NumPy array with shape `(n_samples, n_features)`.
    """
    features = np.asarray(X, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError(
            f"X must be a 2D array of shape (n_samples, n_features); "
            f"got array with shape {features.shape}."
        )

    if features.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    if features.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    if ensure_finite and not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values.")

    return cast(FloatArray, features)


def check_target_vector(
    y: RawArrayLike,
    *,
    n_samples: int | None = None,
    ensure_finite: bool = True,
) -> ArrayLike:
    """Validate target values as a 1D vector.

    Args:
        y: Target vector with shape `(n_samples,)`.
        n_samples: Optional expected number of target values.
        ensure_finite: Whether to reject NaN and infinite values for numeric targets.

    Returns:
        A one-dimensional NumPy array preserving the inferred target dtype.
    """
    targets = np.asarray(y)

    if targets.ndim != 1:
        raise ValueError(
            f"y must be a 1D array of shape (n_samples,); got array with shape {targets.shape}."
        )

    if targets.shape[0] == 0:
        raise ValueError("y must contain at least one target.")

    if n_samples is not None and targets.shape[0] != n_samples:
        raise ValueError(
            f"X and y have inconsistent sample counts: {n_samples} != {targets.shape[0]}."
        )

    if (
        ensure_finite
        and np.issubdtype(targets.dtype, np.number)
        and not np.all(np.isfinite(targets))
    ):
        raise ValueError("y must contain only finite values.")

    return targets


def check_binary_targets(y: RawArrayLike, *, n_samples: int | None = None) -> FloatArray:
    """Validate binary targets encoded as `0` and `1`.

    Args:
        y: Binary target vector with shape `(n_samples,)`.
        n_samples: Optional expected number of labels.

    Returns:
        A float64 vector containing only `0.0` and `1.0`.
    """
    targets = check_target_vector(y, n_samples=n_samples)
    binary_targets = cast(FloatArray, targets.astype(np.float64))
    unique_labels = np.unique(binary_targets)

    if not np.all(np.isin(unique_labels, np.array([0.0, 1.0], dtype=np.float64))):
        raise ValueError("Binary targets must be encoded as 0 and 1.")

    return binary_targets


def check_class_labels(y: RawArrayLike, *, n_samples: int | None = None) -> IntArray:
    """Validate class labels as a 1D integer vector."""
    targets = check_target_vector(y, n_samples=n_samples)

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Class labels must use an integer dtype.")

    return cast(IntArray, targets.astype(np.int_))


def check_sample_weight(sample_weight: RawArrayLike | None, *, n_samples: int) -> FloatArray | None:
    """Validate optional non-negative sample weights.

    Args:
        sample_weight: Optional vector with shape `(n_samples,)`.
        n_samples: Expected number of weights.

    Returns:
        `None` or a float64 vector of non-negative finite weights.
    """
    if sample_weight is None:
        return None

    weights = cast(FloatArray, np.asarray(sample_weight, dtype=np.float64))

    if weights.ndim != 1:
        raise ValueError(
            f"sample_weight must be a 1D array of shape (n_samples,); "
            f"got array with shape {weights.shape}."
        )

    if weights.shape[0] != n_samples:
        raise ValueError(f"sample_weight has length {weights.shape[0]}, but expected {n_samples}.")

    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values.")

    if np.any(weights < 0.0):
        raise ValueError("sample_weight cannot contain negative values.")

    return weights


def check_matching_n_features(X: FloatArray, *, n_features_in: int | None) -> None:
    """Ensure a prediction matrix matches the feature count seen during fitting."""
    if n_features_in is not None and X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features, but this model was fitted with "
            f"{n_features_in} features."
        )
