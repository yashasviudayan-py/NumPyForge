"""Tests for Phase 5 ingestion, training, and evaluation commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.evaluate import evaluate
from pipeline.ingest import ingest
from pipeline.train import train


def test_ingest_creates_processed_dataset(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    output_path = ingest(config_path)

    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as dataset:
        assert dataset["X"].shape == (80, 2)
        assert dataset["y"].shape == (80,)
        labels, counts = np.unique(dataset["y"], return_counts=True)
    np.testing.assert_array_equal(labels, np.array([0, 1]))
    np.testing.assert_array_equal(counts, np.array([40, 40]))


def test_train_writes_artifact_and_metrics(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    result = train(config_path)

    artifact_dir = Path(str(result["artifact_dir"]))
    assert (artifact_dir / "model.npz").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert result["metrics"]["test_accuracy"] >= 0.9
    assert result["metadata"]["training"]["mlflow_run_id"]


def test_evaluate_emits_json_serializable_report(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    evaluation_config = _write_evaluation_config(tmp_path)
    train(config_path)

    result = evaluate(evaluation_config)

    json.dumps(result, allow_nan=False)
    assert result["classification"]["accuracy"] >= 0.9
    assert result["model_version"] == "test"


def _write_config(tmp_path: Path) -> Path:
    config = {
        "artifact": {
            "model_name": "test-model",
            "output_dir": str(tmp_path / "models" / "current"),
            "version": "test",
        },
        "data": {
            "processed_path": str(tmp_path / "data" / "binary_classification.npz"),
            "random_state": 7,
            "test_size": 0.25,
        },
        "mlflow": {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tags": {"phase": "5"},
            "tracking_uri": f"file:{tmp_path / 'mlruns'}",
        },
        "model": {
            "learning_rate": 0.2,
            "n_iterations": 500,
            "penalty": "l2",
            "regularization_strength": 0.01,
            "random_state": 7,
        },
    }
    return _write_json(tmp_path / "training.json", config)


def _write_evaluation_config(tmp_path: Path) -> Path:
    config = {
        "artifact": {"artifact_dir": str(tmp_path / "models" / "current")},
        "data": {
            "processed_path": str(tmp_path / "data" / "binary_classification.npz"),
            "random_state": 7,
            "test_size": 0.25,
        },
    }
    return _write_json(tmp_path / "evaluation.json", config)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path
