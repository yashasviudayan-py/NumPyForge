"""Tests for data splitting and model selection."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.baselines import MajorityClassClassifier, MeanRegressor
from src.linear_model import LinearRegression, LogisticRegression
from src.model_selection import (
    clone_estimator,
    cross_val_score,
    grid_search_cv,
    k_fold_split,
    randomized_search_cv,
    stratified_k_fold_split,
    train_test_split,
)


def test_train_test_split_is_deterministic_with_seed() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.int_)

    first = train_test_split(X, y, test_size=0.3, random_state=42)
    second = train_test_split(X, y, test_size=0.3, random_state=42)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_stratified_train_test_split_preserves_classes() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int_)

    _, _, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)

    assert set(np.asarray(y_train).tolist()) == {0, 1}
    assert set(np.asarray(y_test).tolist()) == {0, 1}


def test_k_fold_covers_each_sample_once_as_validation() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    validation_indices: list[int] = []

    for _, test_indices in k_fold_split(X, n_splits=5):
        validation_indices.extend(int(index) for index in test_indices)

    assert sorted(validation_indices) == list(range(10))


def test_stratified_k_fold_preserves_labels_where_feasible() -> None:
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int_)

    for _, test_indices in stratified_k_fold_split(X, y, n_splits=3, shuffle=True, random_state=3):
        fold_labels = [int(label) for label in y[test_indices]]
        assert set(fold_labels) == {0, 1}


def test_cross_val_score_works_with_models_and_baselines() -> None:
    X_cls = np.array([[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]], dtype=np.float64)
    y_cls = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    X_reg = np.linspace(-1.0, 1.0, 8, dtype=np.float64).reshape(-1, 1)
    y_reg = 1.0 + X_reg[:, 0]

    logistic_scores = cross_val_score(
        LogisticRegression(learning_rate=0.2, n_iterations=200, random_state=4),
        X_cls,
        y_cls,
        cv=4,
        stratified=True,
        random_state=4,
    )
    linear_scores = cross_val_score(LinearRegression(), X_reg, y_reg, cv=4, scoring="r2")
    baseline_scores = cross_val_score(MajorityClassClassifier(), X_cls, y_cls, cv=4)
    mean_scores = cross_val_score(MeanRegressor(), X_reg, y_reg, cv=4, scoring="r2")

    assert logistic_scores.shape == (4,)
    assert linear_scores.shape == (4,)
    assert baseline_scores.shape == (4,)
    assert mean_scores.shape == (4,)


def test_grid_search_selects_best_known_hyperparameter() -> None:
    X = np.array([[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    estimator = LogisticRegression(n_iterations=300, random_state=5)

    result = grid_search_cv(
        estimator,
        {"learning_rate": [0.01, 0.3]},
        X,
        y,
        cv=4,
        stratified=True,
        scoring="accuracy",
        random_state=1,
    )

    assert result.best_params["learning_rate"] == 0.3
    assert result.best_score >= 0.75


def test_randomized_search_is_deterministic_and_clone_does_not_mutate_source() -> None:
    X = np.array([[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    estimator = LogisticRegression(n_iterations=200, random_state=5)

    first = randomized_search_cv(
        estimator,
        {"learning_rate": [0.05, 0.1, 0.2], "regularization_strength": [0.0, 1.0]},
        X,
        y,
        n_iter=3,
        cv=4,
        stratified=True,
        random_state=11,
    )
    second = randomized_search_cv(
        estimator,
        {"learning_rate": [0.05, 0.1, 0.2], "regularization_strength": [0.0, 1.0]},
        X,
        y,
        n_iter=3,
        cv=4,
        stratified=True,
        random_state=11,
    )
    cloned = cast(LogisticRegression, clone_estimator(estimator, {"learning_rate": 0.9}))

    assert first.best_params == second.best_params
    assert estimator.learning_rate != cloned.learning_rate
    assert not estimator.is_fitted
