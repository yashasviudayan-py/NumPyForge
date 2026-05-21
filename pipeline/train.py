"""Training entrypoint for local experiments."""

from __future__ import annotations

import mlflow
import numpy as np

from src.linear_model import LogisticRegression


def main() -> None:
    """Train a starter logistic regression model on a tiny synthetic dataset."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int_)

    model = LogisticRegression(learning_rate=0.1, n_iterations=1_000, random_state=42)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "learning_rate": model.learning_rate,
                "n_iterations": model.n_iterations,
                "penalty": model.penalty,
                "regularization_strength": model.regularization_strength,
            }
        )
        model.fit(X, y)
        accuracy = model.score(X, y)
        mlflow.log_metric("train_accuracy", accuracy)

    print(f"Training accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
