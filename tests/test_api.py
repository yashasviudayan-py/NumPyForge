"""Tests for the FastAPI model-serving application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from src.artifacts import save_logistic_artifact
from src.linear_model import LogisticRegression


def test_health_succeeds_without_artifact(tmp_path: Path) -> None:
    client = TestClient(create_app(_write_serving_config(tmp_path, tmp_path / "missing")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_and_predict_return_503_without_artifact(tmp_path: Path) -> None:
    client = TestClient(create_app(_write_serving_config(tmp_path, tmp_path / "missing")))

    ready = client.get("/ready")
    predict = client.post("/predict", json={"features": [[0.0, 0.0]]})

    assert ready.status_code == 503
    assert ready.json()["detail"]["ready"] is False
    assert predict.status_code == 503


def test_loaded_artifact_serves_metadata_and_predictions(tmp_path: Path) -> None:
    artifact_dir = _create_artifact(tmp_path)
    client = TestClient(create_app(_write_serving_config(tmp_path, artifact_dir)))

    ready = client.get("/ready")
    metadata = client.get("/metadata")
    prediction = client.post("/predict", json={"features": [[0.0, 0.0], [1.0, 1.0]]})

    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert metadata.status_code == 200
    assert metadata.json()["metadata"]["version"] == "api-test"
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["model_version"] == "api-test"
    assert len(body["predictions"]) == 2
    assert len(body["probabilities"]) == 2
    assert body["latency_ms"] >= 0.0


def test_bad_feature_shape_returns_validation_error(tmp_path: Path) -> None:
    artifact_dir = _create_artifact(tmp_path)
    client = TestClient(create_app(_write_serving_config(tmp_path, artifact_dir)))

    response = client.post("/predict", json={"features": [[0.0, 1.0, 2.0]]})

    assert response.status_code == 422
    assert "fitted with 2 features" in response.json()["detail"]


def test_empty_prediction_request_is_rejected(tmp_path: Path) -> None:
    artifact_dir = _create_artifact(tmp_path)
    client = TestClient(create_app(_write_serving_config(tmp_path, artifact_dir)))

    response = client.post("/predict", json={"features": []})

    assert response.status_code == 422


def _create_artifact(tmp_path: Path) -> Path:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int_)
    model = LogisticRegression(learning_rate=0.2, n_iterations=400, random_state=9).fit(X, y)
    artifact_dir = tmp_path / "model"
    save_logistic_artifact(
        model,
        artifact_dir,
        metadata={"version": "api-test"},
        metrics={"accuracy": model.score(X, y)},
    )
    return artifact_dir


def _write_serving_config(tmp_path: Path, artifact_dir: Path) -> Path:
    config = {
        "artifact": {"artifact_dir": str(artifact_dir)},
        "service": {"name": "Test API", "version": "0.1.0"},
    }
    return _write_json(tmp_path / "serving.json", config)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path
