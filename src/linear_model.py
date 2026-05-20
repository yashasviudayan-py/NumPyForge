"""Linear models implemented from first principles with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from src.base import BaseClassifier
from src.math import clip_probabilities, squared_l2_norm, stable_sigmoid
from src.types import FloatArray, IntArray, RawArrayLike
from src.validation import check_binary_targets

Penalty = Literal["l2", None]


@dataclass
class LogisticRegression(BaseClassifier):
    """Binary logistic regression trained with batch gradient descent.

    This class intentionally mirrors the familiar scikit-learn surface while
    keeping the implementation transparent for CS229-style learning.
    """

    learning_rate: float = 0.01
    n_iterations: int = 1_000
    fit_intercept: bool = True
    penalty: Penalty = "l2"
    regularization_strength: float = 0.0
    threshold: float = 0.5
    random_state: int | None = None
    weights_: FloatArray | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        super().__init__(random_state=self.random_state)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        if self.regularization_strength < 0:
            raise ValueError("regularization_strength cannot be negative.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1.")

    def _sigmoid(self, z: FloatArray) -> FloatArray:
        """Compute a numerically stable sigmoid activation."""
        return stable_sigmoid(z)

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> LogisticRegression:
        """Fit logistic regression parameters using batch gradient descent.

        Args:
            X: Training matrix with shape `(n_samples, n_features)`.
            y: Binary labels encoded as `0` and `1`, shape `(n_samples,)`.

        Returns:
            The fitted logistic regression instance.
        """
        features = self._validate_features(X, fitting=True)
        targets = check_binary_targets(y, n_samples=features.shape[0])

        n_samples, n_features = features.shape
        rng = self._rng()
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features).astype(np.float64)
        self.bias_ = 0.0
        self.loss_history_.clear()

        for _ in range(self.n_iterations):
            linear_output = cast(FloatArray, features @ self.weights_ + self.bias_)
            probabilities = self._sigmoid(linear_output)

            error = probabilities - targets
            gradient_w = (features.T @ error) / n_samples
            gradient_b = float(np.sum(error) / n_samples)

            if self.penalty == "l2" and self.regularization_strength > 0:
                gradient_w += (self.regularization_strength / n_samples) * self.weights_

            self.weights_ -= self.learning_rate * gradient_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * gradient_b

            self.loss_history_.append(self._binary_cross_entropy(targets, probabilities))

        self.is_fitted = True
        return self

    def predict_proba(self, X: RawArrayLike) -> FloatArray:
        """Return positive-class probabilities for each sample."""
        self._check_is_fitted()
        features = self._validate_features(X)

        if self.weights_ is None:
            raise RuntimeError("Model weights are unavailable despite fitted state.")

        linear_output = cast(FloatArray, features @ self.weights_ + self.bias_)
        return self._sigmoid(linear_output)

    def predict(self, X: RawArrayLike) -> IntArray:
        """Predict binary class labels using the configured probability threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(np.int_)

    def _binary_cross_entropy(self, y_true: FloatArray, y_pred: FloatArray) -> float:
        """Compute binary cross-entropy with optional L2 regularization."""
        clipped_predictions = clip_probabilities(y_pred)
        data_loss = -np.mean(
            y_true * np.log(clipped_predictions)
            + (1.0 - y_true) * np.log(1.0 - clipped_predictions)
        )

        if self.penalty == "l2" and self.regularization_strength > 0 and self.weights_ is not None:
            l2_loss = (self.regularization_strength / (2.0 * y_true.shape[0])) * float(
                squared_l2_norm(self.weights_)
            )
            return float(data_loss + l2_loss)

        return float(data_loss)
