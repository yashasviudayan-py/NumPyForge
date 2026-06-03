"""Cross-checks that NumPyForge's from-scratch math matches scikit-learn.

These tests are intentionally *not* part of the library's runtime contract:
scikit-learn is never imported by ``numpyforge``. They exist to give an evaluator
independent confidence that the hand-written linear algebra, optimizers, and
metrics agree with a mature reference implementation within tight tolerances.

The whole module is skipped when scikit-learn is not installed, so the core
quality gates remain dependency-light. Install it with ``pip install -e
".[test]"`` (or ``".[dev]"``) to run these.
"""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression as SkLinearRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression as SkLogisticRegression  # noqa: E402
from sklearn.linear_model import Ridge as SkRidge  # noqa: E402
from sklearn.metrics import accuracy_score as sk_accuracy  # noqa: E402
from sklearn.metrics import mean_absolute_error as sk_mae  # noqa: E402
from sklearn.metrics import mean_squared_error as sk_mse  # noqa: E402
from sklearn.metrics import r2_score as sk_r2  # noqa: E402
from sklearn.metrics import roc_auc_score as sk_roc_auc  # noqa: E402

from numpyforge import metrics  # noqa: E402
from numpyforge.linear_model import LinearRegression, LogisticRegression  # noqa: E402


def _regression_data(
    n_samples: int = 120, n_features: int = 5, noise: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coefficients = rng.normal(size=n_features)
    y = X @ coefficients + 0.75 + rng.normal(scale=noise, size=n_samples)
    return X, y


def _classification_data(
    n_samples: int = 240, n_features: int = 4, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coefficients = rng.normal(size=n_features)
    logits = X @ coefficients + 0.3
    labels = (logits > 0.0).astype(np.int_)
    return X, labels


def test_linear_regression_matches_sklearn_ols() -> None:
    X, y = _regression_data()

    ours = LinearRegression().fit(X, y)
    reference = SkLinearRegression().fit(X, y)

    assert ours.weights_ is not None
    np.testing.assert_allclose(ours.weights_, reference.coef_, atol=1e-8)
    assert ours.bias_ == pytest.approx(float(reference.intercept_), abs=1e-8)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), atol=1e-8)


def test_ridge_regression_matches_sklearn_ridge() -> None:
    X, y = _regression_data()
    alpha = 2.5

    ours = LinearRegression(penalty="l2", regularization_strength=alpha).fit(X, y)
    reference = SkRidge(alpha=alpha).fit(X, y)

    assert ours.weights_ is not None
    np.testing.assert_allclose(ours.weights_, reference.coef_, atol=1e-7)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), atol=1e-7)


def test_gradient_descent_linear_regression_converges_to_sklearn() -> None:
    X, y = _regression_data(noise=0.0)

    ours = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.1,
        n_iterations=5_000,
        tol=0.0,
        gradient_tol=1e-10,
    ).fit(X, y)
    reference = SkLinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.predict(X), reference.predict(X), atol=1e-4)


def test_logistic_regression_agrees_with_sklearn() -> None:
    X, y = _classification_data()

    ours = LogisticRegression(
        penalty=None, n_iterations=5_000, learning_rate=0.5, random_state=0
    ).fit(X, y)
    # A very large C disables regularization without relying on deprecated kwargs.
    reference = SkLogisticRegression(C=1e6, max_iter=5_000).fit(X, y)

    agreement = float(np.mean(ours.predict(X) == reference.predict(X)))
    assert agreement >= 0.98

    ours_auc = sk_roc_auc(y, ours.predict_proba(X)[:, 1])
    reference_auc = sk_roc_auc(y, reference.predict_proba(X)[:, 1])
    assert ours_auc == pytest.approx(reference_auc, abs=1e-3)


def test_classification_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=300)
    # Distinct scores avoid tie-handling differences in ROC integration.
    scores = rng.permutation(np.linspace(0.0, 1.0, num=300))
    predictions = (scores >= 0.5).astype(np.int_)

    assert metrics.accuracy_score(labels, predictions) == pytest.approx(
        sk_accuracy(labels, predictions), abs=1e-12
    )
    assert metrics.roc_auc_score(labels, scores) == pytest.approx(
        sk_roc_auc(labels, scores), abs=1e-9
    )


def test_regression_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(11)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.2, size=200)

    assert metrics.r2_score(y_true, y_pred) == pytest.approx(sk_r2(y_true, y_pred), abs=1e-12)
    assert metrics.mean_squared_error(y_true, y_pred) == pytest.approx(
        sk_mse(y_true, y_pred), abs=1e-12
    )
    assert metrics.mean_absolute_error(y_true, y_pred) == pytest.approx(
        sk_mae(y_true, y_pred), abs=1e-12
    )
