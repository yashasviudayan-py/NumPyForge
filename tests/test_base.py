"""Tests for estimator base classes and fitted-state helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.base import BaseClassifier, BaseRegressor
from src.types import FloatArray, IntArray, RawArrayLike


class ConstantClassifier(BaseClassifier):
    """Small test classifier for validating base-class behavior."""

    constant_: int | None

    def __init__(self) -> None:
        super().__init__()
        self.constant_ = None

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> ConstantClassifier:
        features = self._validate_features(X, fitting=True)
        targets = self._validate_targets(y, n_samples=features.shape[0]).astype(np.int_)
        self.constant_ = int(np.bincount(targets).argmax())
        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> IntArray:
        self._check_is_fitted()
        features = self._validate_features(X)

        if self.constant_ is None:
            raise RuntimeError("constant_ is unavailable despite fitted state.")

        return np.full(features.shape[0], self.constant_, dtype=np.int_)


class MeanRegressor(BaseRegressor):
    """Small test regressor for validating the default R2 score."""

    mean_: float | None

    def __init__(self) -> None:
        super().__init__()
        self.mean_ = None

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> MeanRegressor:
        features = self._validate_features(X, fitting=True)
        targets = self._validate_targets(y, n_samples=features.shape[0]).astype(np.float64)
        self.mean_ = float(np.mean(targets))
        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> FloatArray:
        self._check_is_fitted()
        features = self._validate_features(X)

        if self.mean_ is None:
            raise RuntimeError("mean_ is unavailable despite fitted state.")

        return np.full(features.shape[0], self.mean_, dtype=np.float64)


def test_classifier_score_uses_accuracy() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1, 1, 0], dtype=np.int_)

    classifier = ConstantClassifier().fit(X, y)

    assert classifier.score(X, y) == pytest.approx(2.0 / 3.0)


def test_regressor_score_uses_r2() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    regressor = MeanRegressor().fit(X, y)

    assert regressor.score(X, y) == pytest.approx(0.0)


def test_prediction_before_fit_raises_clear_error() -> None:
    classifier = ConstantClassifier()

    with pytest.raises(RuntimeError, match="not fitted yet"):
        classifier.predict(np.array([[1.0]], dtype=np.float64))


def test_feature_count_is_checked_after_fit() -> None:
    classifier = ConstantClassifier().fit(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        np.array([0, 1], dtype=np.int_),
    )

    with pytest.raises(ValueError, match="fitted with 2 features"):
        classifier.predict(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))


def test_parameter_serialization_round_trips_fitted_state(tmp_path: Path) -> None:
    path = tmp_path / "constant-classifier.npz"
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1, 1, 0], dtype=np.int_)
    classifier = ConstantClassifier().fit(X, y)

    classifier.save_parameters(path)
    restored = ConstantClassifier()
    restored.load_parameters(path)

    assert restored.is_fitted
    assert restored.n_features_in_ == 1
    assert restored.constant_ == 1
    np.testing.assert_array_equal(restored.predict(X), classifier.predict(X))
