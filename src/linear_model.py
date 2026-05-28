"""Classical linear models implemented from first principles with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from src.base import BaseClassifier, BaseRegressor
from src.math import (
    binary_cross_entropy,
    categorical_cross_entropy,
    mean_squared_error,
    one_hot,
    softmax,
    stable_sigmoid,
)
from src.optimizers import (
    BatchStrategy,
    GradientDescentConfig,
    OptimizerParameters,
    run_gradient_descent,
)
from src.types import FloatArray, IntArray, RawArrayLike
from src.validation import (
    check_class_weight,
    check_linear_solver_penalty,
    check_multiclass_targets,
    check_penalty,
    check_sample_weight,
)

Penalty = Literal["l1", "l2", None]
MultiClassMode = Literal["auto", "multinomial"]
LinearSolver = Literal["normal_equation", "gradient_descent"]
ClassWeight = dict[int, float] | Literal["balanced"] | None


@dataclass
class LinearRegression(BaseRegressor):
    """Linear regression with closed-form and gradient-descent solvers.

    The model minimizes mean squared error using either the normal equation
    or first-order gradient descent. L2 regularization is supported by both
    solvers; L1 regularization uses a subgradient and therefore requires
    `solver="gradient_descent"`.
    """

    solver: LinearSolver = "normal_equation"
    learning_rate: float = 0.01
    n_iterations: int = 1_000
    fit_intercept: bool = True
    penalty: Penalty = None
    regularization_strength: float = 0.0
    batch_strategy: BatchStrategy = "batch"
    batch_size: int | None = None
    shuffle: bool = True
    tol: float = 1e-6
    gradient_tol: float = 1e-6
    early_stopping: bool = False
    n_iter_no_change: int = 10
    validation_fraction: float = 0.1
    random_state: int | np.random.Generator | None = None
    weights_: FloatArray | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    gradient_norm_history_: list[float] = field(default_factory=list, init=False)
    parameter_norm_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__init__(random_state=self.random_state)
        check_linear_solver_penalty(self.solver, self.penalty)
        _validate_common_training_config(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularization_strength=self.regularization_strength,
            tol=self.tol,
            gradient_tol=self.gradient_tol,
            n_iter_no_change=self.n_iter_no_change,
        )

    def fit(
        self, X: RawArrayLike, y: RawArrayLike, sample_weight: RawArrayLike | None = None
    ) -> LinearRegression:
        """Fit linear regression parameters.

        Args:
            X: Feature matrix with shape `(n_samples, n_features)`.
            y: Regression targets with shape `(n_samples,)`.
            sample_weight: Optional non-negative weights with shape `(n_samples,)`.

        Returns:
            The fitted estimator.
        """
        features = self._validate_features(X, fitting=True)
        targets = cast(
            FloatArray, self._validate_targets(y, n_samples=features.shape[0]).astype(np.float64)
        )
        weights = check_sample_weight(sample_weight, n_samples=features.shape[0])

        if self.solver == "normal_equation":
            self._fit_normal_equation(features, targets, weights)
        else:
            self._fit_gradient_descent(features, targets, weights)

        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> FloatArray:
        """Predict regression values with shape `(n_samples,)`."""
        self._check_is_fitted()
        features = self._validate_features(X)

        if self.weights_ is None:
            raise RuntimeError("Model weights are unavailable despite fitted state.")

        return cast(FloatArray, features @ self.weights_ + self.bias_)

    def _fit_normal_equation(
        self,
        X: FloatArray,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        design = _add_intercept_column(X) if self.fit_intercept else X

        if sample_weight is not None:
            sqrt_weights = np.sqrt(sample_weight)
            weighted_design = design * sqrt_weights[:, np.newaxis]
            weighted_targets = y * sqrt_weights
        else:
            weighted_design = design
            weighted_targets = y

        regularization = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
        if self.penalty == "l2" and self.regularization_strength > 0.0:
            regularization[:] = (
                np.eye(design.shape[1], dtype=np.float64) * self.regularization_strength
            )
            if self.fit_intercept:
                regularization[0, 0] = 0.0

        solution = np.linalg.pinv(weighted_design.T @ weighted_design + regularization) @ (
            weighted_design.T @ weighted_targets
        )

        if self.fit_intercept:
            self.bias_ = float(solution[0])
            self.weights_ = cast(FloatArray, solution[1:].astype(np.float64))
        else:
            self.bias_ = 0.0
            self.weights_ = cast(FloatArray, solution.astype(np.float64))

        self.loss_history_ = [
            self._linear_loss(
                {"weights": self.weights_, "bias": np.array([self.bias_])}, X, y, sample_weight
            )
        ]
        self.validation_loss_history_.clear()
        self.gradient_norm_history_.clear()
        self.parameter_norm_history_ = [_parameter_norm({"weights": self.weights_})]
        self.n_iter_ = 1
        self.converged_ = True

    def _fit_gradient_descent(
        self,
        X: FloatArray,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        parameters: OptimizerParameters = {
            "weights": np.zeros(X.shape[1], dtype=np.float64),
            "bias": np.zeros(1, dtype=np.float64),
        }
        config = self._optimizer_config()
        result = run_gradient_descent(
            initial_parameters=parameters,
            X=X,
            y=y,
            objective=self._linear_loss,
            gradient=self._linear_gradient,
            config=config,
            sample_weight=sample_weight,
        )

        self.weights_ = result.parameters["weights"]
        self.bias_ = float(result.parameters["bias"][0]) if self.fit_intercept else 0.0
        self._copy_optimizer_result(result)

    def _linear_loss(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> float:
        targets = cast(FloatArray, np.asarray(y, dtype=np.float64))
        predictions = cast(FloatArray, X @ parameters["weights"] + parameters["bias"][0])
        loss = 0.5 * mean_squared_error(targets, predictions, sample_weight)
        return loss + _regularization_loss(
            parameters["weights"],
            self.penalty,
            self.regularization_strength,
            _normalizer(sample_weight, X.shape[0]),
        )

    def _linear_gradient(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> OptimizerParameters:
        targets = cast(FloatArray, np.asarray(y, dtype=np.float64))
        predictions = cast(FloatArray, X @ parameters["weights"] + parameters["bias"][0])
        errors = cast(FloatArray, predictions - targets)
        weighted_errors = _apply_sample_weight(errors, sample_weight)
        normalizer = _normalizer(sample_weight, X.shape[0])

        gradient_w = cast(FloatArray, (X.T @ weighted_errors) / normalizer)
        gradient_w += _regularization_gradient(
            parameters["weights"], self.penalty, self.regularization_strength, normalizer
        )
        gradient_b = np.array(
            [float(np.sum(weighted_errors) / normalizer) if self.fit_intercept else 0.0],
            dtype=np.float64,
        )

        return {"weights": gradient_w, "bias": gradient_b}

    def _optimizer_config(self) -> GradientDescentConfig:
        return GradientDescentConfig(
            learning_rate=self.learning_rate,
            max_iter=self.n_iterations,
            batch_strategy=self.batch_strategy,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            tol=self.tol,
            gradient_tol=self.gradient_tol,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

    def _copy_optimizer_result(self, result: object) -> None:
        typed_result = cast("GradientDescentLike", result)
        self.loss_history_ = typed_result.loss_history
        self.validation_loss_history_ = typed_result.validation_loss_history
        self.gradient_norm_history_ = typed_result.gradient_norm_history
        self.parameter_norm_history_ = typed_result.parameter_norm_history
        self.n_iter_ = typed_result.n_iter
        self.converged_ = typed_result.converged


@dataclass
class LogisticRegression(BaseClassifier):
    """Binary and multiclass logistic regression trained with gradient descent."""

    learning_rate: float = 0.01
    n_iterations: int = 1_000
    fit_intercept: bool = True
    penalty: Penalty = "l2"
    regularization_strength: float = 0.0
    threshold: float = 0.5
    multi_class: MultiClassMode = "auto"
    batch_strategy: BatchStrategy = "batch"
    batch_size: int | None = None
    shuffle: bool = True
    tol: float = 1e-6
    gradient_tol: float = 1e-6
    early_stopping: bool = False
    n_iter_no_change: int = 10
    validation_fraction: float = 0.1
    class_weight: ClassWeight = None
    random_state: int | np.random.Generator | None = None
    classes_: IntArray | None = field(default=None, init=False)
    weights_: FloatArray | None = field(default=None, init=False)
    bias_: FloatArray | None = field(default=None, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    gradient_norm_history_: list[float] = field(default_factory=list, init=False)
    parameter_norm_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__init__(random_state=self.random_state)
        check_penalty(self.penalty)
        if self.multi_class not in {"auto", "multinomial"}:
            raise ValueError("multi_class must be 'auto' or 'multinomial'.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        _validate_common_training_config(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularization_strength=self.regularization_strength,
            tol=self.tol,
            gradient_tol=self.gradient_tol,
            n_iter_no_change=self.n_iter_no_change,
        )

    def fit(
        self, X: RawArrayLike, y: RawArrayLike, sample_weight: RawArrayLike | None = None
    ) -> LogisticRegression:
        """Fit binary or multiclass logistic regression.

        Args:
            X: Feature matrix with shape `(n_samples, n_features)`.
            y: Integer class labels with shape `(n_samples,)`.
            sample_weight: Optional non-negative sample weights.

        Returns:
            The fitted estimator.
        """
        features = self._validate_features(X, fitting=True)
        labels = check_multiclass_targets(y, n_samples=features.shape[0])
        self.classes_ = cast(IntArray, np.unique(labels).astype(np.int_))
        encoded_targets = _encode_labels(labels, self.classes_)
        weights = self._effective_sample_weight(labels, sample_weight)

        if self._uses_binary_objective():
            self._fit_binary(features, encoded_targets, weights)
        else:
            self._fit_multiclass(features, encoded_targets, weights)

        self.is_fitted = True
        return self

    def predict_proba(self, X: RawArrayLike) -> FloatArray:
        """Return class probabilities with shape `(n_samples, n_classes)`."""
        self._check_is_fitted()
        features = self._validate_features(X)

        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Model parameters are unavailable despite fitted state.")

        if self._uses_binary_objective():
            logits = cast(FloatArray, features @ self.weights_ + self.bias_[0])
            positive_probability = stable_sigmoid(logits)
            return cast(
                FloatArray,
                np.column_stack((1.0 - positive_probability, positive_probability)),
            )

        logits = cast(FloatArray, features @ self.weights_ + self.bias_)
        return softmax(logits, axis=1)

    def predict(self, X: RawArrayLike) -> IntArray:
        """Predict class labels with shape `(n_samples,)`."""
        probabilities = self.predict_proba(X)

        if self.classes_ is None:
            raise RuntimeError("Class labels are unavailable despite fitted state.")

        if self._uses_binary_objective():
            encoded = (probabilities[:, 1] >= self.threshold).astype(np.int_)
        else:
            encoded = cast(IntArray, np.argmax(probabilities, axis=1).astype(np.int_))

        return cast(IntArray, self.classes_[encoded])

    def _fit_binary(
        self,
        X: FloatArray,
        y_encoded: IntArray,
        sample_weight: FloatArray | None,
    ) -> None:
        parameters: OptimizerParameters = {
            "weights": cast(
                FloatArray,
                np.asarray(
                    self._rng().normal(loc=0.0, scale=0.01, size=X.shape[1]),
                    dtype=np.float64,
                ),
            ),
            "bias": np.zeros(1, dtype=np.float64),
        }
        y_float = cast(FloatArray, y_encoded.astype(np.float64))
        result = run_gradient_descent(
            initial_parameters=parameters,
            X=X,
            y=y_float,
            objective=self._binary_loss,
            gradient=self._binary_gradient,
            config=self._optimizer_config(),
            sample_weight=sample_weight,
        )

        self.weights_ = result.parameters["weights"]
        self.bias_ = result.parameters["bias"]
        self._copy_optimizer_result(result)

    def _fit_multiclass(
        self,
        X: FloatArray,
        y_encoded: IntArray,
        sample_weight: FloatArray | None,
    ) -> None:
        if self.classes_ is None:
            raise RuntimeError("Classes are unavailable during fitting.")

        n_classes = self.classes_.shape[0]
        parameters: OptimizerParameters = {
            "weights": self._rng()
            .normal(loc=0.0, scale=0.01, size=(X.shape[1], n_classes))
            .astype(np.float64),
            "bias": np.zeros(n_classes, dtype=np.float64),
        }
        result = run_gradient_descent(
            initial_parameters=parameters,
            X=X,
            y=y_encoded,
            objective=self._multiclass_loss,
            gradient=self._multiclass_gradient,
            config=self._optimizer_config(),
            sample_weight=sample_weight,
        )

        self.weights_ = result.parameters["weights"]
        self.bias_ = result.parameters["bias"]
        self._copy_optimizer_result(result)

    def _binary_loss(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> float:
        targets = cast(FloatArray, np.asarray(y, dtype=np.float64))
        probabilities = stable_sigmoid(
            cast(FloatArray, X @ parameters["weights"] + parameters["bias"][0])
        )
        loss = binary_cross_entropy(targets, probabilities, sample_weight)
        return loss + _regularization_loss(
            parameters["weights"],
            self.penalty,
            self.regularization_strength,
            _normalizer(sample_weight, X.shape[0]),
        )

    def _binary_gradient(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> OptimizerParameters:
        targets = cast(FloatArray, np.asarray(y, dtype=np.float64))
        probabilities = stable_sigmoid(
            cast(FloatArray, X @ parameters["weights"] + parameters["bias"][0])
        )
        errors = cast(FloatArray, probabilities - targets)
        weighted_errors = _apply_sample_weight(errors, sample_weight)
        normalizer = _normalizer(sample_weight, X.shape[0])

        gradient_w = cast(FloatArray, (X.T @ weighted_errors) / normalizer)
        gradient_w += _regularization_gradient(
            parameters["weights"], self.penalty, self.regularization_strength, normalizer
        )
        gradient_b = np.array(
            [float(np.sum(weighted_errors) / normalizer) if self.fit_intercept else 0.0],
            dtype=np.float64,
        )
        return {"weights": gradient_w, "bias": gradient_b}

    def _multiclass_loss(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> float:
        targets = cast(IntArray, np.asarray(y, dtype=np.int_))
        probabilities = softmax(
            cast(FloatArray, X @ parameters["weights"] + parameters["bias"]), axis=1
        )
        target_matrix = one_hot(targets, n_classes=probabilities.shape[1])
        loss = categorical_cross_entropy(target_matrix, probabilities, sample_weight)
        return loss + _regularization_loss(
            parameters["weights"],
            self.penalty,
            self.regularization_strength,
            _normalizer(sample_weight, X.shape[0]),
        )

    def _multiclass_gradient(
        self,
        parameters: OptimizerParameters,
        X: FloatArray,
        y: RawArrayLike,
        sample_weight: FloatArray | None,
    ) -> OptimizerParameters:
        targets = cast(IntArray, np.asarray(y, dtype=np.int_))
        probabilities = softmax(
            cast(FloatArray, X @ parameters["weights"] + parameters["bias"]), axis=1
        )
        target_matrix = one_hot(targets, n_classes=probabilities.shape[1])
        errors = cast(FloatArray, probabilities - target_matrix)
        weighted_errors = _apply_sample_weight_matrix(errors, sample_weight)
        normalizer = _normalizer(sample_weight, X.shape[0])

        gradient_w = cast(FloatArray, (X.T @ weighted_errors) / normalizer)
        gradient_w += _regularization_gradient(
            parameters["weights"], self.penalty, self.regularization_strength, normalizer
        )
        if self.fit_intercept:
            gradient_b = cast(FloatArray, np.sum(weighted_errors, axis=0) / normalizer)
        else:
            gradient_b = np.zeros(parameters["bias"].shape, dtype=np.float64)
        return {"weights": gradient_w, "bias": gradient_b}

    def _effective_sample_weight(
        self,
        labels: IntArray,
        sample_weight: RawArrayLike | None,
    ) -> FloatArray | None:
        weights = check_sample_weight(sample_weight, n_samples=labels.shape[0])
        if self.classes_ is None:
            raise RuntimeError("Classes are unavailable during fitting.")

        class_weights = check_class_weight(self.class_weight, classes=self.classes_, y=labels)
        class_weight_vector = np.array(
            [class_weights[int(label)] for label in labels], dtype=np.float64
        )

        if weights is None and np.all(class_weight_vector == 1.0):
            return None

        if weights is None:
            return class_weight_vector

        return cast(FloatArray, weights * class_weight_vector)

    def _uses_binary_objective(self) -> bool:
        if self.classes_ is None:
            return self.multi_class != "multinomial"
        return self.multi_class == "auto" and self.classes_.shape[0] == 2

    def _optimizer_config(self) -> GradientDescentConfig:
        return GradientDescentConfig(
            learning_rate=self.learning_rate,
            max_iter=self.n_iterations,
            batch_strategy=self.batch_strategy,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            tol=self.tol,
            gradient_tol=self.gradient_tol,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

    def _copy_optimizer_result(self, result: object) -> None:
        typed_result = cast("GradientDescentLike", result)
        self.loss_history_ = typed_result.loss_history
        self.validation_loss_history_ = typed_result.validation_loss_history
        self.gradient_norm_history_ = typed_result.gradient_norm_history
        self.parameter_norm_history_ = typed_result.parameter_norm_history
        self.n_iter_ = typed_result.n_iter
        self.converged_ = typed_result.converged


class GradientDescentLike:
    """Structural helper for copying optimizer histories without importing at runtime."""

    loss_history: list[float]
    validation_loss_history: list[float]
    gradient_norm_history: list[float]
    parameter_norm_history: list[float]
    n_iter: int
    converged: bool


def _validate_common_training_config(
    *,
    learning_rate: float,
    n_iterations: int,
    regularization_strength: float,
    tol: float,
    gradient_tol: float,
    n_iter_no_change: int,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if regularization_strength < 0.0:
        raise ValueError("regularization_strength cannot be negative.")
    if tol < 0.0:
        raise ValueError("tol cannot be negative.")
    if gradient_tol < 0.0:
        raise ValueError("gradient_tol cannot be negative.")
    if n_iter_no_change <= 0:
        raise ValueError("n_iter_no_change must be positive.")


def _add_intercept_column(X: FloatArray) -> FloatArray:
    return cast(FloatArray, np.column_stack((np.ones(X.shape[0], dtype=np.float64), X)))


def _encode_labels(labels: IntArray, classes: IntArray) -> IntArray:
    mapping = {int(label): index for index, label in enumerate(classes)}
    return np.array([mapping[int(label)] for label in labels], dtype=np.int_)


def _normalizer(sample_weight: FloatArray | None, n_samples: int) -> float:
    if sample_weight is None:
        return float(n_samples)

    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        raise ValueError("sample weights must sum to a positive value.")

    return weight_sum


def _apply_sample_weight(values: FloatArray, sample_weight: FloatArray | None) -> FloatArray:
    if sample_weight is None:
        return values

    return cast(FloatArray, values * sample_weight)


def _apply_sample_weight_matrix(values: FloatArray, sample_weight: FloatArray | None) -> FloatArray:
    if sample_weight is None:
        return values

    return cast(FloatArray, values * sample_weight[:, np.newaxis])


def _regularization_loss(
    weights: FloatArray,
    penalty: Penalty,
    regularization_strength: float,
    normalizer: float,
) -> float:
    if penalty is None or regularization_strength == 0.0:
        return 0.0

    if penalty == "l2":
        return regularization_strength * float(np.sum(weights * weights)) / (2.0 * normalizer)

    return regularization_strength * float(np.sum(np.abs(weights))) / normalizer


def _regularization_gradient(
    weights: FloatArray,
    penalty: Penalty,
    regularization_strength: float,
    normalizer: float,
) -> FloatArray:
    if penalty is None or regularization_strength == 0.0:
        return np.zeros_like(weights)

    if penalty == "l2":
        return cast(FloatArray, (regularization_strength / normalizer) * weights)

    return cast(FloatArray, (regularization_strength / normalizer) * np.sign(weights))


def _parameter_norm(parameters: OptimizerParameters) -> float:
    total = 0.0
    for value in parameters.values():
        total += float(np.sum(value * value))
    return float(np.sqrt(total))
