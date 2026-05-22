"""Scikit-like multilayer perceptron estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from src.base import BaseClassifier, BaseRegressor
from src.math import one_hot
from src.neural_network.initializers import Initializer
from src.neural_network.layers import (
    Dense,
    Dropout,
    Layer,
    LeakyReLUActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
    TanhActivation,
)
from src.neural_network.losses import CategoricalCrossEntropyLoss, MeanSquaredErrorLoss, NeuralLoss
from src.neural_network.network import SequentialNetwork
from src.optimizers import (
    AdamOptimizer,
    LearningRateSchedule,
    MomentumOptimizer,
    Optimizer,
    OptimizerParameters,
    RMSPropOptimizer,
    ScheduleKind,
    SGDOptimizer,
)
from src.types import FloatArray, IntArray, RandomState, RawArrayLike
from src.validation import check_multiclass_targets, check_validation_fraction

ActivationName = Literal["relu", "leaky_relu", "sigmoid", "tanh"]
OptimizerName = Literal["sgd", "momentum", "rmsprop", "adam"]


@dataclass
class MLPClassifier(BaseClassifier):
    """Multilayer perceptron classifier trained with backpropagation."""

    hidden_layer_sizes: tuple[int, ...] = (100,)
    activation: ActivationName = "relu"
    optimizer: OptimizerName = "adam"
    learning_rate: float = 0.001
    learning_rate_schedule: ScheduleKind = "constant"
    max_iter: int = 200
    batch_size: int | None = None
    shuffle: bool = True
    dropout_rate: float = 0.0
    alpha: float = 0.0
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-6
    initializer: Initializer = "xavier"
    random_state: RandomState = None
    classes_: IntArray | None = field(default=None, init=False)
    network_: SequentialNetwork | None = field(default=None, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    learning_rate_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__init__(random_state=self.random_state)
        _validate_mlp_config(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            dropout_rate=self.dropout_rate,
            alpha=self.alpha,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
        )

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> MLPClassifier:
        """Fit the MLP classifier."""
        features = self._validate_features(X, fitting=True)
        labels = check_multiclass_targets(y, n_samples=features.shape[0])
        self.classes_ = cast(IntArray, np.unique(labels).astype(np.int_))
        encoded = _encode_labels(labels, self.classes_)
        targets = one_hot(encoded, n_classes=self.classes_.shape[0])

        rng = self._rng()
        self.network_ = _build_network(
            n_inputs=features.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_outputs=self.classes_.shape[0],
            activation=self.activation,
            output_activation=SoftmaxActivation(),
            dropout_rate=self.dropout_rate,
            initializer=self.initializer,
            rng=rng,
        )
        _train_network(
            estimator=self,
            network=self.network_,
            X=features,
            y=targets,
            loss=CategoricalCrossEntropyLoss(),
            rng=rng,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: RawArrayLike) -> FloatArray:
        """Return class probabilities with shape `(n_samples, n_classes)`."""
        self._check_is_fitted()
        if self.network_ is None:
            raise RuntimeError("Network is unavailable despite fitted state.")
        return self.network_.forward(self._validate_features(X), training=False)

    def predict(self, X: RawArrayLike) -> IntArray:
        """Predict class labels with shape `(n_samples,)`."""
        if self.classes_ is None:
            raise RuntimeError("Class labels are unavailable despite fitted state.")
        encoded = cast(IntArray, np.argmax(self.predict_proba(X), axis=1).astype(np.int_))
        return cast(IntArray, self.classes_[encoded])


@dataclass
class MLPRegressor(BaseRegressor):
    """Multilayer perceptron regressor trained with backpropagation."""

    hidden_layer_sizes: tuple[int, ...] = (100,)
    activation: ActivationName = "relu"
    optimizer: OptimizerName = "adam"
    learning_rate: float = 0.001
    learning_rate_schedule: ScheduleKind = "constant"
    max_iter: int = 200
    batch_size: int | None = None
    shuffle: bool = True
    dropout_rate: float = 0.0
    alpha: float = 0.0
    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-6
    initializer: Initializer = "xavier"
    random_state: RandomState = None
    network_: SequentialNetwork | None = field(default=None, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    learning_rate_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__init__(random_state=self.random_state)
        _validate_mlp_config(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            dropout_rate=self.dropout_rate,
            alpha=self.alpha,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
        )

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> MLPRegressor:
        """Fit the MLP regressor."""
        features = self._validate_features(X, fitting=True)
        targets = cast(
            FloatArray, self._validate_targets(y, n_samples=features.shape[0]).astype(np.float64)
        ).reshape(-1, 1)

        rng = self._rng()
        self.network_ = _build_network(
            n_inputs=features.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_outputs=1,
            activation=self.activation,
            output_activation=None,
            dropout_rate=self.dropout_rate,
            initializer=self.initializer,
            rng=rng,
        )
        _train_network(
            estimator=self,
            network=self.network_,
            X=features,
            y=targets,
            loss=MeanSquaredErrorLoss(),
            rng=rng,
        )
        self.is_fitted = True
        return self

    def predict(self, X: RawArrayLike) -> FloatArray:
        """Predict regression values with shape `(n_samples,)`."""
        self._check_is_fitted()
        if self.network_ is None:
            raise RuntimeError("Network is unavailable despite fitted state.")
        predictions = self.network_.forward(self._validate_features(X), training=False)
        return cast(FloatArray, predictions.ravel())


def _validate_mlp_config(
    *,
    hidden_layer_sizes: tuple[int, ...],
    activation: ActivationName,
    optimizer: OptimizerName,
    learning_rate: float,
    max_iter: int,
    dropout_rate: float,
    alpha: float,
    validation_fraction: float,
    n_iter_no_change: int,
    tol: float,
) -> None:
    if any(size <= 0 for size in hidden_layer_sizes):
        raise ValueError("hidden_layer_sizes must contain only positive layer sizes.")
    if activation not in {"relu", "leaky_relu", "sigmoid", "tanh"}:
        raise ValueError("activation must be 'relu', 'leaky_relu', 'sigmoid', or 'tanh'.")
    if optimizer not in {"sgd", "momentum", "rmsprop", "adam"}:
        raise ValueError("optimizer must be 'sgd', 'momentum', 'rmsprop', or 'adam'.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0.0, 1.0).")
    if alpha < 0.0:
        raise ValueError("alpha cannot be negative.")
    if n_iter_no_change <= 0:
        raise ValueError("n_iter_no_change must be positive.")
    if tol < 0.0:
        raise ValueError("tol cannot be negative.")
    check_validation_fraction(validation_fraction)


def _build_network(
    *,
    n_inputs: int,
    hidden_layer_sizes: tuple[int, ...],
    n_outputs: int,
    activation: ActivationName,
    output_activation: Layer | None,
    dropout_rate: float,
    initializer: Initializer,
    rng: np.random.Generator,
) -> SequentialNetwork:
    layers: list[Layer] = []
    current_width = n_inputs

    for hidden_width in hidden_layer_sizes:
        layers.append(Dense(current_width, hidden_width, initializer=initializer, rng=rng))
        layers.append(_activation_layer(activation))
        if dropout_rate > 0.0:
            layers.append(Dropout(dropout_rate, rng=rng))
        current_width = hidden_width

    layers.append(Dense(current_width, n_outputs, initializer=initializer, rng=rng))
    if output_activation is not None:
        layers.append(output_activation)

    return SequentialNetwork(layers)


def _activation_layer(name: ActivationName) -> Layer:
    if name == "relu":
        return ReLUActivation()
    if name == "leaky_relu":
        return LeakyReLUActivation()
    if name == "sigmoid":
        return SigmoidActivation()
    return TanhActivation()


def _train_network(
    *,
    estimator: MLPClassifier | MLPRegressor,
    network: SequentialNetwork,
    X: FloatArray,
    y: FloatArray,
    loss: NeuralLoss,
    rng: np.random.Generator,
) -> None:
    optimizer = _make_optimizer(estimator.optimizer)
    schedule = LearningRateSchedule(
        initial_learning_rate=estimator.learning_rate,
        kind=estimator.learning_rate_schedule,
        max_iter=estimator.max_iter,
    )
    train_indices, validation_indices = _split_train_validation_indices(
        X.shape[0], estimator.early_stopping, estimator.validation_fraction, rng
    )
    batch_size = _effective_batch_size(estimator.batch_size, train_indices.shape[0])

    estimator.loss_history_.clear()
    estimator.validation_loss_history_.clear()
    estimator.learning_rate_history_.clear()
    estimator.converged_ = False
    best_loss = float("inf")
    no_improvement_count = 0

    for iteration in range(estimator.max_iter):
        epoch_indices = train_indices.copy()
        if estimator.shuffle:
            rng.shuffle(epoch_indices)

        learning_rate = schedule.value(iteration)
        for start in range(0, epoch_indices.shape[0], batch_size):
            batch_indices = epoch_indices[start : start + batch_size]
            predictions = network.forward(X[batch_indices], training=True)
            gradient = loss.gradient(y[batch_indices], predictions)
            network.backward(gradient)
            gradients = network.grads()
            _add_l2_weight_decay(
                gradients, network.params(), estimator.alpha, batch_indices.shape[0]
            )
            optimizer.step(network.params(), gradients, learning_rate=learning_rate)

        train_loss = _network_loss(
            network, X[train_indices], y[train_indices], loss, estimator.alpha
        )
        estimator.loss_history_.append(train_loss)
        estimator.learning_rate_history_.append(learning_rate)

        if validation_indices is not None:
            monitored_loss = _network_loss(
                network, X[validation_indices], y[validation_indices], loss, estimator.alpha
            )
            estimator.validation_loss_history_.append(monitored_loss)
        else:
            monitored_loss = train_loss

        estimator.n_iter_ = iteration + 1
        if best_loss - monitored_loss > estimator.tol:
            best_loss = monitored_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= estimator.n_iter_no_change:
            estimator.converged_ = True
            break


def _make_optimizer(name: OptimizerName) -> Optimizer:
    if name == "sgd":
        return SGDOptimizer()
    if name == "momentum":
        return MomentumOptimizer()
    if name == "rmsprop":
        return RMSPropOptimizer()
    return AdamOptimizer()


def _network_loss(
    network: SequentialNetwork,
    X: FloatArray,
    y: FloatArray,
    loss: NeuralLoss,
    alpha: float,
) -> float:
    predictions = network.forward(X, training=False)
    return loss.value(y, predictions) + _l2_penalty(network.params(), alpha, X.shape[0])


def _l2_penalty(parameters: OptimizerParameters, alpha: float, n_samples: int) -> float:
    if alpha == 0.0:
        return 0.0

    total = 0.0
    for name, value in parameters.items():
        if name.endswith(".weights"):
            total += float(np.sum(value * value))
    return alpha * total / (2.0 * n_samples)


def _add_l2_weight_decay(
    gradients: OptimizerParameters,
    parameters: OptimizerParameters,
    alpha: float,
    n_samples: int,
) -> None:
    if alpha == 0.0:
        return

    for name, value in parameters.items():
        if name.endswith(".weights"):
            gradients[name] += (alpha / n_samples) * value


def _effective_batch_size(batch_size: int | None, n_samples: int) -> int:
    if batch_size is None:
        return min(32, n_samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if batch_size > n_samples:
        raise ValueError("batch_size cannot exceed the number of samples.")
    return batch_size


def _split_train_validation_indices(
    n_samples: int,
    early_stopping: bool,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.int_]],
    np.ndarray[tuple[int, ...], np.dtype[np.int_]] | None,
]:
    indices = np.arange(n_samples, dtype=np.int_)
    if not early_stopping or validation_fraction == 0.0:
        return indices, None

    if n_samples < 2:
        raise ValueError("early_stopping with validation_fraction requires at least two samples.")

    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_validation = max(1, int(np.ceil(n_samples * validation_fraction)))
    n_validation = min(n_validation, n_samples - 1)
    return shuffled[n_validation:], shuffled[:n_validation]


def _encode_labels(labels: IntArray, classes: IntArray) -> IntArray:
    mapping = {int(label): index for index, label in enumerate(classes)}
    return np.array([mapping[int(label)] for label in labels], dtype=np.int_)
