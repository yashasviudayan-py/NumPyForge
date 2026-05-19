"""Shared abstractions for NumPy-based machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
ArrayLike: TypeAlias = NDArray[Any]


class SupportsFitPredict(Protocol):
    """Protocol for estimators exposing a scikit-learn-like interface."""

    is_fitted: bool

    def fit(self, X: FloatArray, y: ArrayLike) -> SupportsFitPredict:
        """Fit the estimator on feature matrix `X` and target vector `y`."""

    def predict(self, X: FloatArray) -> ArrayLike:
        """Generate predictions for feature matrix `X`."""


class BaseModel(ABC):
    """Abstract base class for all custom NumPy ML models.

    Subclasses should implement `fit` and `predict`, set `is_fitted` to `True`
    after successful training, and use `_validate_features` /
    `_validate_targets` for consistent input handling.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.n_features_in_: int | None = None

    @abstractmethod
    def fit(self, X: FloatArray, y: ArrayLike) -> BaseModel:
        """Fit model parameters from training data.

        Args:
            X: Two-dimensional training matrix with shape `(n_samples, n_features)`.
            y: Target values with shape `(n_samples,)`.

        Returns:
            The fitted model instance.
        """

    @abstractmethod
    def predict(self, X: FloatArray) -> ArrayLike:
        """Predict target values for the provided feature matrix.

        Args:
            X: Two-dimensional feature matrix with shape `(n_samples, n_features)`.

        Returns:
            Model predictions.
        """

    def _validate_features(self, X: FloatArray, *, fitting: bool = False) -> FloatArray:
        """Validate and coerce a feature matrix to a two-dimensional float array."""
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

        if fitting:
            self.n_features_in_ = int(features.shape[1])
        elif self.n_features_in_ is not None and features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {features.shape[1]} features, but this model was fitted "
                f"with {self.n_features_in_} features."
            )

        return features

    def _validate_targets(self, y: ArrayLike, *, n_samples: int) -> ArrayLike:
        """Validate and coerce a target vector."""
        targets = np.asarray(y)

        if targets.ndim != 1:
            raise ValueError(
                f"y must be a 1D array of shape (n_samples,); "
                f"got array with shape {targets.shape}."
            )

        if targets.shape[0] != n_samples:
            raise ValueError(
                f"X and y have inconsistent sample counts: " f"{n_samples} != {targets.shape[0]}."
            )

        return targets

    def _check_is_fitted(self) -> None:
        """Raise an error if prediction is attempted before fitting."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call `.fit(X, y)` before prediction."
            )
