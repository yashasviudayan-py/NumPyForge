"""Data splitting, cross-validation, and model-search helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from typing import Protocol, cast

import numpy as np

from src.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from src.random import check_random_state
from src.types import ArrayLike, FloatArray, IntArray, RandomState, RawArrayLike
from src.validation import check_feature_matrix, check_target_vector


class EstimatorProtocol(Protocol):
    """Minimal estimator interface used by model-selection helpers."""

    def fit(self, X: RawArrayLike, y: RawArrayLike) -> object:
        """Fit estimator."""

    def predict(self, X: RawArrayLike) -> ArrayLike:
        """Predict target values."""

    def score(self, X: RawArrayLike, y: RawArrayLike) -> float:
        """Return estimator score."""


Scoring = str | Callable[[EstimatorProtocol, FloatArray, ArrayLike], float] | None


@dataclass(frozen=True)
class SearchResult:
    """Result returned by grid and randomized search."""

    best_estimator: EstimatorProtocol
    best_params: dict[str, object]
    best_score: float
    cv_results: list[dict[str, object]]


def train_test_split(
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    test_size: float | int = 0.25,
    shuffle: bool = True,
    stratify: RawArrayLike | None = None,
    random_state: RandomState = None,
) -> tuple[FloatArray, FloatArray, ArrayLike, ArrayLike]:
    """Split arrays into train and test partitions."""
    features = check_feature_matrix(X)
    targets = check_target_vector(y, n_samples=features.shape[0])
    n_samples = features.shape[0]
    n_test = _resolve_test_size(test_size, n_samples=n_samples)
    rng = check_random_state(random_state)

    if stratify is not None:
        stratify_labels = cast(
            IntArray, check_target_vector(stratify, n_samples=n_samples).astype(np.int_)
        )
        test_indices = _stratified_test_indices(stratify_labels, n_test=n_test, rng=rng)
    else:
        indices = np.arange(n_samples, dtype=np.int_)
        if shuffle:
            rng.shuffle(indices)
        test_indices = indices[:n_test]

    test_mask = np.zeros(n_samples, dtype=np.bool_)
    test_mask[test_indices] = True
    train_indices = np.arange(n_samples, dtype=np.int_)[~test_mask]
    sorted_test_indices = np.sort(test_indices)

    return (
        features[train_indices],
        features[sorted_test_indices],
        targets[train_indices],
        targets[sorted_test_indices],
    )


def k_fold_split(
    X: RawArrayLike,
    y: RawArrayLike | None = None,
    *,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: RandomState = None,
) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield train/test index pairs for K-fold cross-validation."""
    features = check_feature_matrix(X)
    if y is not None:
        check_target_vector(y, n_samples=features.shape[0])
    yield from _k_fold_indices(
        features.shape[0], n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )


def stratified_k_fold_split(
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: RandomState = None,
) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield stratified train/test index pairs for classification labels."""
    features = check_feature_matrix(X)
    labels = cast(IntArray, check_target_vector(y, n_samples=features.shape[0]).astype(np.int_))
    _validate_n_splits(n_splits, features.shape[0])
    rng = check_random_state(random_state)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label).astype(np.int_)
        if shuffle:
            rng.shuffle(label_indices)
        for offset, index in enumerate(label_indices):
            folds[offset % n_splits].append(int(index))

    all_indices = np.arange(features.shape[0], dtype=np.int_)
    for fold in folds:
        test_indices = np.array(sorted(fold), dtype=np.int_)
        train_mask = np.ones(features.shape[0], dtype=np.bool_)
        train_mask[test_indices] = False
        yield all_indices[train_mask], test_indices


def cross_val_score(
    estimator: EstimatorProtocol,
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    cv: int = 5,
    scoring: Scoring = None,
    stratified: bool = False,
    random_state: RandomState = None,
) -> FloatArray:
    """Return cross-validation scores for an estimator."""
    features = check_feature_matrix(X)
    targets = check_target_vector(y, n_samples=features.shape[0])
    scorer = _resolve_scorer(scoring)
    splitter: Iterable[tuple[IntArray, IntArray]]
    if stratified:
        splitter = stratified_k_fold_split(
            features, targets, n_splits=cv, shuffle=True, random_state=random_state
        )
    else:
        splitter = k_fold_split(
            features, targets, n_splits=cv, shuffle=True, random_state=random_state
        )

    scores: list[float] = []
    for train_indices, test_indices in splitter:
        candidate = clone_estimator(estimator)
        candidate.fit(features[train_indices], targets[train_indices])
        scores.append(scorer(candidate, features[test_indices], targets[test_indices]))

    return np.array(scores, dtype=np.float64)


def grid_search_cv(
    estimator: EstimatorProtocol,
    param_grid: dict[str, list[object]],
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    cv: int = 5,
    scoring: Scoring = None,
    stratified: bool = False,
    random_state: RandomState = None,
) -> SearchResult:
    """Evaluate every parameter combination and return the best result."""
    combinations = _parameter_grid(param_grid)
    return _search_cv(
        estimator,
        combinations,
        X,
        y,
        cv=cv,
        scoring=scoring,
        stratified=stratified,
        random_state=random_state,
    )


def randomized_search_cv(
    estimator: EstimatorProtocol,
    param_distributions: dict[str, list[object]],
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    n_iter: int = 10,
    cv: int = 5,
    scoring: Scoring = None,
    stratified: bool = False,
    random_state: RandomState = None,
) -> SearchResult:
    """Evaluate a deterministic random subset of finite parameter candidates."""
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    combinations = _parameter_grid(param_distributions)
    if n_iter > len(combinations):
        raise ValueError("n_iter cannot exceed the number of finite parameter combinations.")
    rng = check_random_state(random_state)
    selected = rng.choice(len(combinations), size=n_iter, replace=False)
    sampled = [combinations[int(index)] for index in selected]
    return _search_cv(
        estimator,
        sampled,
        X,
        y,
        cv=cv,
        scoring=scoring,
        stratified=stratified,
        random_state=random_state,
    )


def clone_estimator(
    estimator: EstimatorProtocol, params: dict[str, object] | None = None
) -> EstimatorProtocol:
    """Clone an estimator from dataclass init fields or constructor defaults."""
    overrides = {} if params is None else params
    if is_dataclass(estimator):
        init_params = {
            field.name: getattr(estimator, field.name)
            for field in fields(estimator)
            if field.init and hasattr(estimator, field.name)
        }
    else:
        init_params = {}
    init_params.update(overrides)
    return estimator.__class__(**init_params)


def _search_cv(
    estimator: EstimatorProtocol,
    candidates: list[dict[str, object]],
    X: RawArrayLike,
    y: RawArrayLike,
    *,
    cv: int,
    scoring: Scoring,
    stratified: bool,
    random_state: RandomState,
) -> SearchResult:
    if not candidates:
        raise ValueError("At least one parameter candidate is required.")
    features = check_feature_matrix(X)
    targets = check_target_vector(y, n_samples=features.shape[0])
    cv_results: list[dict[str, object]] = []
    best_score = -float("inf")
    best_params: dict[str, object] = {}
    best_estimator: EstimatorProtocol | None = None

    for params in candidates:
        candidate = clone_estimator(estimator, params)
        scores = cross_val_score(
            candidate,
            features,
            targets,
            cv=cv,
            scoring=scoring,
            stratified=stratified,
            random_state=random_state,
        )
        mean_score = float(np.mean(scores))
        cv_results.append(
            {
                "params": params,
                "mean_score": mean_score,
                "scores": scores.tolist(),
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_estimator = clone_estimator(estimator, params)

    if best_estimator is None:
        raise RuntimeError("Search failed to select an estimator.")

    best_estimator.fit(features, targets)
    return SearchResult(
        best_estimator=best_estimator,
        best_params=best_params,
        best_score=best_score,
        cv_results=cv_results,
    )


def _resolve_scorer(
    scoring: Scoring,
) -> Callable[[EstimatorProtocol, FloatArray, ArrayLike], float]:
    if scoring is None:
        return lambda estimator, X, y: float(estimator.score(X, y))
    if callable(scoring):
        return scoring

    scorers: dict[str, Callable[[EstimatorProtocol, FloatArray, ArrayLike], float]] = {
        "accuracy": lambda estimator, X, y: accuracy_score(y, estimator.predict(X)),
        "f1": lambda estimator, X, y: f1_score(y, estimator.predict(X)),
        "r2": lambda estimator, X, y: r2_score(y, estimator.predict(X)),
        "neg_mean_squared_error": lambda estimator, X, y: -mean_squared_error(
            y, estimator.predict(X)
        ),
        "neg_root_mean_squared_error": lambda estimator, X, y: -root_mean_squared_error(
            y, estimator.predict(X)
        ),
        "neg_mean_absolute_error": lambda estimator, X, y: -mean_absolute_error(
            y, estimator.predict(X)
        ),
    }
    if scoring not in scorers:
        raise ValueError(f"Unknown scoring value: {scoring}.")
    return scorers[scoring]


def _parameter_grid(parameter_values: dict[str, list[object]]) -> list[dict[str, object]]:
    if not parameter_values:
        raise ValueError("parameter grid cannot be empty.")
    for name, values in parameter_values.items():
        if not values:
            raise ValueError(f"parameter grid for {name!r} cannot be empty.")
    keys = list(parameter_values)
    return [
        dict(zip(keys, values, strict=True))
        for values in product(*(parameter_values[key] for key in keys))
    ]


def _resolve_test_size(test_size: float | int, *, n_samples: int) -> int:
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("float test_size must be in (0.0, 1.0).")
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)
    if not 0 < n_test < n_samples:
        raise ValueError("test_size must leave at least one train and one test sample.")
    return n_test


def _stratified_test_indices(
    labels: IntArray, *, n_test: int, rng: np.random.Generator
) -> IntArray:
    test_indices: list[int] = []
    classes = np.unique(labels)
    for label in classes:
        label_indices = np.flatnonzero(labels == label).astype(np.int_)
        rng.shuffle(label_indices)
        class_test_count = max(1, int(round(n_test * label_indices.shape[0] / labels.shape[0])))
        class_test_count = min(class_test_count, label_indices.shape[0] - 1)
        test_indices.extend(int(index) for index in label_indices[:class_test_count])
    while len(test_indices) > n_test:
        test_indices.pop()
    if len(test_indices) < n_test:
        remaining = [index for index in range(labels.shape[0]) if index not in set(test_indices)]
        rng.shuffle(remaining)
        test_indices.extend(remaining[: n_test - len(test_indices)])
    return np.array(test_indices, dtype=np.int_)


def _k_fold_indices(
    n_samples: int,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: RandomState,
) -> Iterator[tuple[IntArray, IntArray]]:
    _validate_n_splits(n_splits, n_samples)
    indices = np.arange(n_samples, dtype=np.int_)
    if shuffle:
        check_random_state(random_state).shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int_)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:])).astype(np.int_)
        yield cast(IntArray, train_indices), cast(IntArray, test_indices)
        current = stop


def _validate_n_splits(n_splits: int, n_samples: int) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed the number of samples.")
