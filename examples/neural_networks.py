"""Phase 3 examples for pure NumPy neural networks.

Run from the project root:

    python examples/neural_networks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_network import MLPClassifier, MLPRegressor  # noqa: E402


def xor_demo() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int_)
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        optimizer="adam",
        learning_rate=0.05,
        max_iter=1_000,
        batch_size=4,
        tol=0.0,
        n_iter_no_change=1_000,
        random_state=42,
    ).fit(X, y)
    print("XOR classification")
    print(f"  predictions={model.predict(X).tolist()}")
    print(f"  probabilities={model.predict_proba(X).round(3).tolist()}")


def multiclass_blob_demo() -> None:
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
    model = MLPClassifier(
        hidden_layer_sizes=(6,),
        activation="tanh",
        optimizer="adam",
        learning_rate=0.03,
        max_iter=800,
        batch_size=6,
        random_state=7,
    ).fit(X, y)
    print("Multiclass blobs")
    print(f"  predictions={model.predict(X).tolist()}")


def regression_curve_demo() -> None:
    X = np.linspace(-1.0, 1.0, 30, dtype=np.float64).reshape(-1, 1)
    y = X[:, 0] ** 2
    model = MLPRegressor(
        hidden_layer_sizes=(12,),
        activation="tanh",
        optimizer="adam",
        learning_rate=0.03,
        max_iter=600,
        batch_size=10,
        tol=0.0,
        n_iter_no_change=600,
        random_state=9,
    ).fit(X, y)
    mse = float(np.mean((model.predict(X) - y) ** 2))
    print("Nonlinear regression")
    print(f"  mse={mse:.6f}")


def main() -> None:
    xor_demo()
    multiclass_blob_demo()
    regression_curve_demo()


if __name__ == "__main__":
    main()
