"""Tests for versioned model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import artifact_status, load_logistic_artifact, save_logistic_artifact
from src.linear_model import LogisticRegression


def test_logistic_artifact_round_trips_predictions(tmp_path: Path) -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int_)
    model = LogisticRegression(learning_rate=0.2, n_iterations=400, random_state=7).fit(X, y)

    saved = save_logistic_artifact(
        model,
        tmp_path / "model",
        metadata={"version": "test-version", "model_name": "test-model"},
        metrics={"accuracy": model.score(X, y)},
    )
    loaded = load_logistic_artifact(saved.path)

    assert loaded.metadata["version"] == "test-version"
    assert loaded.metrics["accuracy"] == pytest.approx(1.0)
    np.testing.assert_array_equal(loaded.model.predict(X), model.predict(X))
    np.testing.assert_allclose(loaded.model.predict_proba(X), model.predict_proba(X))


def test_artifact_status_reports_missing_directory(tmp_path: Path) -> None:
    status = artifact_status(tmp_path / "missing")

    assert not status.ready
    assert "does not exist" in str(status.error)


def test_artifact_status_reports_corrupt_metadata(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "corrupt"
    artifact_dir.mkdir()
    (artifact_dir / "metadata.json").write_text("{not-json", encoding="utf-8")
    (artifact_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model.npz").write_bytes(b"not-a-model")

    status = artifact_status(artifact_dir)

    assert not status.ready
    assert status.error is not None


def test_artifact_loader_rejects_wrong_schema_version(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bad-schema"
    artifact_dir.mkdir()
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 999,
                "model_type": "binary_logistic_regression",
                "hyperparameters": {},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "model.npz").write_bytes(b"not-a-model")

    with pytest.raises(ValueError, match="Unsupported artifact schema version"):
        load_logistic_artifact(artifact_dir)


def test_loaded_artifact_validates_feature_count(tmp_path: Path) -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.2, 1.0], [-1.0, -1.0]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int_)
    model = LogisticRegression(learning_rate=0.2, n_iterations=400, random_state=7).fit(X, y)
    saved = save_logistic_artifact(
        model,
        tmp_path / "model",
        metadata={"version": "test-version"},
        metrics={"accuracy": model.score(X, y)},
    )

    loaded = load_logistic_artifact(saved.path)
    with pytest.raises(ValueError, match="fitted with 2 features"):
        loaded.model.predict(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
