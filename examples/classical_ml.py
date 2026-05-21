"""Phase 2 examples for classical ML models implemented with pure NumPy.

Run from the project root:

    python examples/classical_ml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.linear_model import LinearRegression, LogisticRegression  # noqa: E402


def linear_regression_closed_form_demo() -> None:
    X = np.linspace(-2.0, 2.0, 20, dtype=np.float64).reshape(-1, 1)
    y = 1.25 + 2.5 * X[:, 0]
    model = LinearRegression().fit(X, y)
    print("Closed-form linear regression")
    print(f"  bias={model.bias_:.3f}, weight={model.weights_[0]:.3f}")


def linear_regression_gradient_descent_demo() -> None:
    X = np.linspace(-1.0, 1.0, 30, dtype=np.float64).reshape(-1, 1)
    y = -0.5 + 4.0 * X[:, 0]
    model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.2,
        n_iterations=1_000,
        gradient_tol=1e-8,
    ).fit(X, y)
    print("Gradient-descent linear regression")
    final_loss = model.loss_history_[-1]
    print(
        f"  n_iter={model.n_iter_}, converged={model.converged_}, " f"final_loss={final_loss:.6f}"
    )


def binary_logistic_regression_demo() -> None:
    X = np.array([[0.0], [0.5], [1.0], [2.0], [2.5], [3.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)
    model = LogisticRegression(learning_rate=0.4, n_iterations=500, random_state=42).fit(X, y)
    print("Binary logistic regression")
    print(f"  predictions={model.predict(X).tolist()}")
    print(f"  positive probabilities={model.predict_proba(X)[:, 1].round(3).tolist()}")


def multiclass_logistic_regression_demo() -> None:
    X = np.array(
        [
            [-2.0, -2.0],
            [-1.5, -2.0],
            [2.0, -2.0],
            [2.5, -1.5],
            [0.0, 2.0],
            [0.5, 2.5],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int_)
    model = LogisticRegression(
        learning_rate=0.5,
        n_iterations=1_000,
        multi_class="multinomial",
        random_state=7,
    ).fit(X, y)
    print("Multiclass softmax regression")
    print(f"  predictions={model.predict(X).tolist()}")


def regularization_demo() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    unregularized = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.05,
        n_iterations=800,
        penalty=None,
    ).fit(X, y)
    l2_model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.05,
        n_iterations=800,
        penalty="l2",
        regularization_strength=4.0,
    ).fit(X, y)
    print("Regularization effect")
    print(
        "  |w| unregularized="
        f"{abs(unregularized.weights_[0]):.3f}, |w| l2={abs(l2_model.weights_[0]):.3f}"
    )


def main() -> None:
    linear_regression_closed_form_demo()
    linear_regression_gradient_descent_demo()
    binary_logistic_regression_demo()
    multiclass_logistic_regression_demo()
    regularization_demo()


if __name__ == "__main__":
    main()
