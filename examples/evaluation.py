"""Phase 4 examples for validation, metrics, and model selection.

Run from the project root:

    python examples/evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.metrics as metrics  # noqa: E402
import src.model_selection as model_selection  # noqa: E402
from src.baselines import MajorityClassClassifier, MeanRegressor  # noqa: E402
from src.linear_model import LinearRegression, LogisticRegression  # noqa: E402


def classification_demo() -> None:
    X = np.array([[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = LogisticRegression(learning_rate=0.3, n_iterations=600, random_state=3).fit(
        X_train, y_train
    )
    baseline = MajorityClassClassifier().fit(X_train, y_train)
    report = metrics.classification_report_dict(
        y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]
    )
    print("Classification evaluation")
    print(f"  accuracy={report['accuracy']:.3f}")
    print(f"  baseline_accuracy={metrics.accuracy_score(y_test, baseline.predict(X_test)):.3f}")
    cv_scores = model_selection.cross_val_score(model, X, y, cv=4, stratified=True, random_state=1)
    print(f"  cv_scores={cv_scores.round(3).tolist()}")


def search_demo() -> None:
    X = np.array([[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    estimator = LogisticRegression(n_iterations=400, random_state=5)
    grid_result = model_selection.grid_search_cv(
        estimator,
        {"learning_rate": [0.05, 0.2], "regularization_strength": [0.0, 1.0]},
        X,
        y,
        cv=4,
        scoring="accuracy",
        stratified=True,
        random_state=11,
    )
    random_result = model_selection.randomized_search_cv(
        estimator,
        {"learning_rate": [0.05, 0.1, 0.2], "regularization_strength": [0.0, 1.0]},
        X,
        y,
        n_iter=3,
        cv=4,
        scoring="accuracy",
        stratified=True,
        random_state=11,
    )
    print("Model selection")
    print(f"  grid_best={grid_result.best_params}, score={grid_result.best_score:.3f}")
    print(f"  random_best={random_result.best_params}, score={random_result.best_score:.3f}")


def regression_demo() -> None:
    X = np.linspace(-2.0, 2.0, 20, dtype=np.float64).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    model = LinearRegression().fit(X_train, y_train)
    baseline = MeanRegressor().fit(X_train, y_train)
    report = metrics.regression_report_dict(
        y_test, model.predict(X_test), n_features=X_test.shape[1]
    )
    baseline_report = metrics.regression_report_dict(
        y_test, baseline.predict(X_test), n_features=X_test.shape[1]
    )
    print("Regression evaluation")
    print(f"  r2={report['r2']:.3f}")
    print(f"  baseline_r2={baseline_report['r2']:.3f}")


def main() -> None:
    classification_demo()
    search_demo()
    regression_demo()


if __name__ == "__main__":
    main()
