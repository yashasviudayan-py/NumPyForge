"""Shared estimator abstractions for NumPy-based machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Self, cast

import numpy as np

from src.random import check_random_state
from src.types import (
    ArrayLike,
    FloatArray,
    ParameterState,
    PathLikeString,
    RandomState,
    RawArrayLike,
)
from src.validation import (
    check_feature_matrix,
    check_matching_n_features,
    check_target_vector,
)


class SupportsFitPredict(Protocol):
    """Protocol for estimators exposing a scikit-learn-like interface."""

    is_fitted: bool

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> Self:
        """Fit the estimator on feature matrix `X` and target vector `y`."""

    def predict(self, X: RawArrayLike) -> ArrayLike:
        """Generate predictions for feature matrix `X`."""


class LossFunction(Protocol):
    """Protocol for prediction-space losses.

    Implementations receive targets with shape `(n_samples,)` or
    `(n_samples, n_outputs)` and predictions with matching leading dimension.
    """

    def value(self, y_true: ArrayLike, y_pred: FloatArray) -> float:
        """Return the scalar loss value."""

    def gradient(self, y_true: ArrayLike, y_pred: FloatArray) -> FloatArray:
        """Return the gradient of the loss with respect to predictions."""


class ObjectiveFunction(Protocol):
    """Protocol for optimization objectives over model parameters."""

    def value(self, parameters: ParameterState, X: FloatArray, y: ArrayLike) -> float:
        """Return the scalar objective for the provided parameters and data."""

    def gradient(self, parameters: ParameterState, X: FloatArray, y: ArrayLike) -> ParameterState:
        """Return parameter-wise gradients for the objective."""


class BaseEstimator(ABC):
    """Base class for all NumPyForge estimators.

    Estimators accept feature matrices with shape `(n_samples, n_features)` and
    target vectors with shape `(n_samples,)`. Subclasses should call
    `_validate_features(..., fitting=True)` during `fit` and
    `_validate_features(...)` during prediction to enforce consistent feature
    dimensions.
    """

    def __init__(self, *, random_state: RandomState = None) -> None:
        self.is_fitted: bool = False
        self.n_features_in_: int | None = None
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: RawArrayLike, y: RawArrayLike) -> Self:
        """Fit model parameters from training data."""

    @abstractmethod
    def predict(self, X: RawArrayLike) -> ArrayLike:
        """Predict target values for a feature matrix."""

    @abstractmethod
    def score(self, X: RawArrayLike, y: RawArrayLike) -> float:
        """Return a default quality score for the estimator."""

    def _rng(self) -> np.random.Generator:
        """Return a NumPy random generator using this estimator's random state."""
        return check_random_state(self.random_state)

    def _validate_features(self, X: RawArrayLike, *, fitting: bool = False) -> FloatArray:
        """Validate a feature matrix and enforce fitted feature count.

        Args:
            X: Feature matrix with shape `(n_samples, n_features)`.
            fitting: Whether the estimator is currently fitting.

        Returns:
            A float64 feature matrix with shape `(n_samples, n_features)`.
        """
        features = check_feature_matrix(X)

        if fitting:
            self.n_features_in_ = int(features.shape[1])
        else:
            check_matching_n_features(features, n_features_in=self.n_features_in_)

        return features

    def _validate_targets(self, y: RawArrayLike, *, n_samples: int) -> ArrayLike:
        """Validate a target vector with shape `(n_samples,)`."""
        return check_target_vector(y, n_samples=n_samples)

    def _check_is_fitted(self) -> None:
        """Raise an error if prediction is attempted before fitting."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call `.fit(X, y)` before prediction."
            )

    def state_dict(self) -> ParameterState:
        """Return fitted parameters and metadata as a serializable dictionary.

        Attributes ending in `_` are treated as fitted state, mirroring the
        scikit-learn convention. `None` values are omitted.
        """
        state: ParameterState = {}

        for name, value in self.__dict__.items():
            if not name.endswith("_") or value is None:
                continue

            if isinstance(value, np.ndarray):
                state[name] = value
            elif isinstance(value, np.generic):
                state[name] = cast(float | int | bool, value.item())
            elif isinstance(value, int | float | bool):
                state[name] = value

        return state

    def load_state_dict(self, state: ParameterState) -> None:
        """Load fitted parameters and metadata from a state dictionary."""
        for name, value in state.items():
            setattr(self, name, value)

        self.is_fitted = True

    def save_parameters(self, path: PathLikeString) -> None:
        """Persist fitted parameters to a NumPy `.npz` archive."""
        self._check_is_fitted()
        archive_state: dict[str, object] = dict(self.state_dict())
        np.savez(path, **archive_state)  # type: ignore[arg-type]

    def load_parameters(self, path: PathLikeString) -> None:
        """Load fitted parameters from a NumPy `.npz` archive."""
        with np.load(path, allow_pickle=False) as archive:
            state: ParameterState = {}
            for name in archive.files:
                value = archive[name]
                state[name] = value.item() if value.ndim == 0 else value

        self.load_state_dict(state)


class BaseClassifier(BaseEstimator):
    """Base class for supervised classifiers."""

    def score(self, X: RawArrayLike, y: RawArrayLike) -> float:
        """Return classification accuracy for labels with shape `(n_samples,)`."""
        targets = check_target_vector(y)
        predictions = self.predict(X)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions have shape {predictions.shape}, but targets have shape "
                f"{targets.shape}."
            )

        return float(np.mean(predictions == targets))


class BaseRegressor(BaseEstimator):
    """Base class for supervised regressors."""

    def score(self, X: RawArrayLike, y: RawArrayLike) -> float:
        """Return coefficient of determination, R2."""
        targets = cast(FloatArray, check_target_vector(y).astype(np.float64))
        predictions = cast(FloatArray, np.asarray(self.predict(X), dtype=np.float64))

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions have shape {predictions.shape}, but targets have shape "
                f"{targets.shape}."
            )

        residual_sum_of_squares = float(np.sum((targets - predictions) ** 2))
        total_sum_of_squares = float(np.sum((targets - np.mean(targets)) ** 2))

        if total_sum_of_squares == 0.0:
            return 1.0 if residual_sum_of_squares == 0.0 else 0.0

        return 1.0 - residual_sum_of_squares / total_sum_of_squares


BaseModel = BaseEstimator
