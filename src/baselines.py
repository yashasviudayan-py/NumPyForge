"""Simple deterministic baseline estimators."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.base import BaseClassifier, BaseRegressor
from src.types import FloatArray, IntArray, RawArrayLike
from src.validation import check_class_labels


class MajorityClassClassifier(BaseClassifier):
    """Classifier that always predicts the most frequent training class."""

    def __init__(self) -> None:
        super().__init__()
        self.classes_: IntArray | None = None
        self.majority_class_: int | None = None

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> MajorityClassClassifier:
        features = self._validate_features(X, fitting=True)
        labels = check_class_labels(y, n_samples=features.shape[0])
        classes, counts = np.unique(labels, return_counts=True)
        self.classes_ = cast(IntArray, classes.astype(np.int_))
        self.majority_class_ = int(classes[np.argmax(counts)])
        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> IntArray:
        self._check_is_fitted()
        features = self._validate_features(X)
        if self.majority_class_ is None:
            raise RuntimeError("majority_class_ is unavailable despite fitted state.")
        return np.full(features.shape[0], self.majority_class_, dtype=np.int_)


class MeanRegressor(BaseRegressor):
    """Regressor that always predicts the training-target mean."""

    def __init__(self) -> None:
        super().__init__()
        self.mean_: float | None = None

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> MeanRegressor:
        features = self._validate_features(X, fitting=True)
        targets = cast(
            FloatArray, self._validate_targets(y, n_samples=features.shape[0]).astype(np.float64)
        )
        self.mean_ = float(np.mean(targets))
        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> FloatArray:
        self._check_is_fitted()
        features = self._validate_features(X)
        if self.mean_ is None:
            raise RuntimeError("mean_ is unavailable despite fitted state.")
        return np.full(features.shape[0], self.mean_, dtype=np.float64)
