"""Reusable optimization routines for NumPyForge estimators."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np

from src.random import check_random_state
from src.types import ArrayLike, FloatArray, RandomState
from src.validation import check_batch_size, check_validation_fraction

BatchStrategy = Literal["batch", "mini_batch", "stochastic"]
ScheduleKind = Literal["constant", "step", "cosine"]
OptimizerParameters = dict[str, FloatArray]
ObjectiveCallback = Callable[[OptimizerParameters, FloatArray, ArrayLike, FloatArray | None], float]
GradientCallback = Callable[
    [OptimizerParameters, FloatArray, ArrayLike, FloatArray | None], OptimizerParameters
]


class Optimizer(Protocol):
    """Protocol for stateful parameter-update optimizers."""

    def step(
        self,
        parameters: OptimizerParameters,
        gradients: OptimizerParameters,
        *,
        learning_rate: float,
    ) -> None:
        """Update parameters in place."""


class SGDOptimizer:
    """Plain stochastic gradient descent."""

    def step(
        self,
        parameters: OptimizerParameters,
        gradients: OptimizerParameters,
        *,
        learning_rate: float,
    ) -> None:
        for name, gradient in gradients.items():
            parameters[name] -= learning_rate * gradient


@dataclass
class MomentumOptimizer:
    """SGD with momentum."""

    momentum: float = 0.9
    velocity_: OptimizerParameters | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0.0, 1.0).")

    def step(
        self,
        parameters: OptimizerParameters,
        gradients: OptimizerParameters,
        *,
        learning_rate: float,
    ) -> None:
        if self.velocity_ is None:
            self.velocity_ = _zeros_like(parameters)

        for name, gradient in gradients.items():
            self.velocity_[name] = self.momentum * self.velocity_[name] - learning_rate * gradient
            parameters[name] += self.velocity_[name]


@dataclass
class RMSPropOptimizer:
    """RMSProp optimizer with exponential moving average of squared gradients."""

    decay_rate: float = 0.9
    eps: float = 1e-8
    squared_average_: OptimizerParameters | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay_rate < 1.0:
            raise ValueError("decay_rate must be in [0.0, 1.0).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def step(
        self,
        parameters: OptimizerParameters,
        gradients: OptimizerParameters,
        *,
        learning_rate: float,
    ) -> None:
        if self.squared_average_ is None:
            self.squared_average_ = _zeros_like(parameters)

        for name, gradient in gradients.items():
            self.squared_average_[name] = (
                self.decay_rate * self.squared_average_[name]
                + (1.0 - self.decay_rate) * gradient * gradient
            )
            parameters[name] -= (
                learning_rate * gradient / (np.sqrt(self.squared_average_[name]) + self.eps)
            )


@dataclass
class AdamOptimizer:
    """Adam optimizer with bias-corrected first and second moments."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    first_moment_: OptimizerParameters | None = None
    second_moment_: OptimizerParameters | None = None
    timestep_: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in [0.0, 1.0).")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in [0.0, 1.0).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def step(
        self,
        parameters: OptimizerParameters,
        gradients: OptimizerParameters,
        *,
        learning_rate: float,
    ) -> None:
        if self.first_moment_ is None:
            self.first_moment_ = _zeros_like(parameters)
        if self.second_moment_ is None:
            self.second_moment_ = _zeros_like(parameters)

        self.timestep_ += 1
        for name, gradient in gradients.items():
            self.first_moment_[name] = (
                self.beta1 * self.first_moment_[name] + (1.0 - self.beta1) * gradient
            )
            self.second_moment_[name] = (
                self.beta2 * self.second_moment_[name] + (1.0 - self.beta2) * gradient * gradient
            )
            first_unbiased = self.first_moment_[name] / (1.0 - self.beta1**self.timestep_)
            second_unbiased = self.second_moment_[name] / (1.0 - self.beta2**self.timestep_)
            parameters[name] -= (
                learning_rate * first_unbiased / (np.sqrt(second_unbiased) + self.eps)
            )


@dataclass(frozen=True)
class LearningRateSchedule:
    """Learning-rate schedule for iterative optimization."""

    initial_learning_rate: float
    kind: ScheduleKind = "constant"
    step_size: int = 10
    decay_rate: float = 0.5
    max_iter: int = 1

    def __post_init__(self) -> None:
        if self.initial_learning_rate <= 0.0:
            raise ValueError("initial_learning_rate must be positive.")
        if self.kind not in {"constant", "step", "cosine"}:
            raise ValueError("kind must be 'constant', 'step', or 'cosine'.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError("decay_rate must be in (0.0, 1.0].")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

    def value(self, iteration: int) -> float:
        """Return the learning rate for a zero-based iteration index."""
        if iteration < 0:
            raise ValueError("iteration cannot be negative.")

        if self.kind == "constant":
            return self.initial_learning_rate

        if self.kind == "step":
            return self.initial_learning_rate * self.decay_rate ** (iteration // self.step_size)

        progress = min(iteration, self.max_iter) / self.max_iter
        return self.initial_learning_rate * 0.5 * (1.0 + float(np.cos(np.pi * progress)))


@dataclass(frozen=True)
class GradientDescentConfig:
    """Configuration for first-order gradient descent."""

    learning_rate: float = 0.01
    max_iter: int = 1_000
    batch_strategy: BatchStrategy = "batch"
    batch_size: int | None = None
    shuffle: bool = True
    tol: float = 1e-6
    gradient_tol: float = 1e-6
    early_stopping: bool = False
    n_iter_no_change: int = 10
    validation_fraction: float = 0.1
    random_state: RandomState = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.batch_strategy not in {"batch", "mini_batch", "stochastic"}:
            raise ValueError("batch_strategy must be 'batch', 'mini_batch', or 'stochastic'.")
        if self.tol < 0.0:
            raise ValueError("tol cannot be negative.")
        if self.gradient_tol < 0.0:
            raise ValueError("gradient_tol cannot be negative.")
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive.")
        check_validation_fraction(self.validation_fraction)


@dataclass(frozen=True)
class GradientDescentResult:
    """Result returned by `run_gradient_descent`."""

    parameters: OptimizerParameters
    loss_history: list[float]
    validation_loss_history: list[float]
    gradient_norm_history: list[float]
    parameter_norm_history: list[float]
    n_iter: int
    converged: bool


def run_gradient_descent(
    *,
    initial_parameters: OptimizerParameters,
    X: FloatArray,
    y: ArrayLike,
    objective: ObjectiveCallback,
    gradient: GradientCallback,
    config: GradientDescentConfig,
    sample_weight: FloatArray | None = None,
) -> GradientDescentResult:
    """Optimize parameters with batch, mini-batch, or stochastic gradient descent."""
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("Gradient descent requires at least one sample.")

    parameters = _copy_parameters(initial_parameters)
    rng = check_random_state(config.random_state)
    train_indices, validation_indices = _split_train_validation_indices(n_samples, config, rng)

    check_batch_size(
        _effective_batch_size(config, train_indices.shape[0]), n_samples=train_indices.shape[0]
    )

    loss_history: list[float] = []
    validation_loss_history: list[float] = []
    gradient_norm_history: list[float] = []
    parameter_norm_history: list[float] = []
    converged = False
    best_loss = float("inf")
    no_improvement_count = 0

    for iteration in range(1, config.max_iter + 1):
        last_gradient: OptimizerParameters | None = None
        for batch_indices in _iter_batches(train_indices, config, rng):
            batch_gradient = gradient(
                parameters,
                X[batch_indices],
                y[batch_indices],
                None if sample_weight is None else sample_weight[batch_indices],
            )
            _apply_gradient_step(parameters, batch_gradient, learning_rate=config.learning_rate)
            last_gradient = batch_gradient

        train_loss = objective(
            parameters,
            X[train_indices],
            y[train_indices],
            None if sample_weight is None else sample_weight[train_indices],
        )
        loss_history.append(train_loss)

        if validation_indices is not None:
            validation_loss = objective(
                parameters,
                X[validation_indices],
                y[validation_indices],
                None if sample_weight is None else sample_weight[validation_indices],
            )
            validation_loss_history.append(validation_loss)
            monitored_loss = validation_loss
        else:
            monitored_loss = train_loss

        if last_gradient is None:
            last_gradient = gradient(
                parameters,
                X[train_indices],
                y[train_indices],
                None if sample_weight is None else sample_weight[train_indices],
            )

        gradient_norm = _parameter_norm(last_gradient)
        parameter_norm = _parameter_norm(parameters)
        gradient_norm_history.append(gradient_norm)
        parameter_norm_history.append(parameter_norm)

        if gradient_norm <= config.gradient_tol:
            converged = True
            return GradientDescentResult(
                parameters=parameters,
                loss_history=loss_history,
                validation_loss_history=validation_loss_history,
                gradient_norm_history=gradient_norm_history,
                parameter_norm_history=parameter_norm_history,
                n_iter=iteration,
                converged=converged,
            )

        if best_loss - monitored_loss > config.tol:
            best_loss = monitored_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= config.n_iter_no_change:
            converged = True
            return GradientDescentResult(
                parameters=parameters,
                loss_history=loss_history,
                validation_loss_history=validation_loss_history,
                gradient_norm_history=gradient_norm_history,
                parameter_norm_history=parameter_norm_history,
                n_iter=iteration,
                converged=converged,
            )

    return GradientDescentResult(
        parameters=parameters,
        loss_history=loss_history,
        validation_loss_history=validation_loss_history,
        gradient_norm_history=gradient_norm_history,
        parameter_norm_history=parameter_norm_history,
        n_iter=config.max_iter,
        converged=converged,
    )


def _copy_parameters(parameters: OptimizerParameters) -> OptimizerParameters:
    return {name: cast(FloatArray, value.copy()) for name, value in parameters.items()}


def _zeros_like(parameters: OptimizerParameters) -> OptimizerParameters:
    return {name: np.zeros_like(value, dtype=np.float64) for name, value in parameters.items()}


def _apply_gradient_step(
    parameters: OptimizerParameters,
    gradient: OptimizerParameters,
    *,
    learning_rate: float,
) -> None:
    for name, value in gradient.items():
        parameters[name] -= learning_rate * value


def _parameter_norm(parameters: OptimizerParameters) -> float:
    total = 0.0
    for value in parameters.values():
        total += float(np.sum(value * value))
    return float(np.sqrt(total))


def _effective_batch_size(config: GradientDescentConfig, n_samples: int) -> int | None:
    if config.batch_strategy == "batch":
        return n_samples
    if config.batch_strategy == "stochastic":
        return 1
    return config.batch_size if config.batch_size is not None else min(32, n_samples)


def _iter_batches(
    indices: np.ndarray[tuple[int, ...], np.dtype[np.int_]],
    config: GradientDescentConfig,
    rng: np.random.Generator,
) -> Iterator[np.ndarray[tuple[int, ...], np.dtype[np.int_]]]:
    batch_size = _effective_batch_size(config, indices.shape[0])
    if batch_size is None:
        raise RuntimeError("Unable to determine batch size.")

    epoch_indices = indices.copy()
    if config.shuffle and config.batch_strategy != "batch":
        rng.shuffle(epoch_indices)

    for start in range(0, epoch_indices.shape[0], batch_size):
        yield epoch_indices[start : start + batch_size]


def _split_train_validation_indices(
    n_samples: int,
    config: GradientDescentConfig,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.int_]],
    np.ndarray[tuple[int, ...], np.dtype[np.int_]] | None,
]:
    indices = np.arange(n_samples, dtype=np.int_)

    if not config.early_stopping or config.validation_fraction == 0.0:
        return indices, None

    if n_samples < 2:
        raise ValueError("early_stopping with validation_fraction requires at least two samples.")

    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_validation = max(1, int(np.ceil(n_samples * config.validation_fraction)))
    n_validation = min(n_validation, n_samples - 1)

    return shuffled[n_validation:], shuffled[:n_validation]
