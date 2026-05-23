"""Local evaluation entrypoint for synthetic NumPyForge examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from src.metrics import classification_report_dict, regression_report_dict  # noqa: E402
from src.model_selection import train_test_split  # noqa: E402


def main() -> None:
    """Run compact classification and regression evaluation examples."""
    X_cls = np.array(
        [[0.0], [0.2], [0.4], [0.8], [1.2], [1.8], [2.2], [2.8]],
        dtype=np.float64,
    )
    y_cls = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int_)
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.25, stratify=y_cls, random_state=42
    )
    classifier = LogisticRegression(learning_rate=0.3, n_iterations=600, random_state=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_score = classifier.predict_proba(X_test)[:, 1]

    X_reg = np.linspace(-2.0, 2.0, 20, dtype=np.float64).reshape(-1, 1)
    y_reg = 1.0 + 2.0 * X_reg[:, 0]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=7
    )
    regressor = LinearRegression().fit(Xr_train, yr_train)

    print(
        json.dumps(
            {
                "classification": classification_report_dict(y_test, y_pred, y_score),
                "regression": regression_report_dict(
                    yr_test, regressor.predict(Xr_test), n_features=Xr_test.shape[1]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
