"""Developer example: the standard NumPyForge estimator lifecycle.

Run from the project root:

    python examples/estimator_lifecycle.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.linear_model import LogisticRegression  # noqa: E402


def main() -> None:
    """Fit, score, serialize, restore, and predict with a NumPyForge estimator."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1], dtype=np.int_)

    model = LogisticRegression(learning_rate=0.1, n_iterations=2_000, random_state=42)
    model.fit(X, y)

    print(f"Accuracy: {model.score(X, y):.3f}")
    print(f"Probabilities: {model.predict_proba(X).round(3).tolist()}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        parameter_path = Path(tmp_dir) / "logistic-regression.npz"
        model.save_parameters(parameter_path)

        restored = LogisticRegression()
        restored.load_parameters(parameter_path)
        print(f"Restored predictions: {restored.predict(X).tolist()}")


if __name__ == "__main__":
    main()
