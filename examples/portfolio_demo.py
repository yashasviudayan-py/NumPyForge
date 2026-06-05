"""Backend/MLOps portfolio demo for NumPyForge.

Run from the project root:

    python examples/portfolio_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numpyforge.artifacts import load_logistic_artifact  # noqa: E402
from pipeline.evaluate import evaluate  # noqa: E402
from pipeline.ingest import ingest  # noqa: E402
from pipeline.train import train  # noqa: E402


def main() -> None:
    """Run the end-to-end portfolio demo and print a compact summary."""
    with tempfile.TemporaryDirectory(prefix="numpyforge-demo-") as workspace:
        workspace_path = Path(workspace)
        training_config = _write_training_config(workspace_path)
        evaluation_config = _write_evaluation_config(workspace_path)
        serving_config = _write_serving_config(workspace_path)

        processed_path = ingest(training_config)
        train_result = train(training_config)
        evaluation = evaluate(evaluation_config)
        artifact = load_logistic_artifact(Path(str(train_result["artifact_dir"])))

        with _demo_serving_environment(serving_config):
            from api.main import create_app  # noqa: PLC0415

            demo_app = create_app(serving_config)
        client = TestClient(demo_app)
        health = client.get("/health")
        ready = client.get("/ready")
        metadata = client.get("/metadata")
        prediction = client.post("/predict", json={"features": [[0.0, 0.0], [1.0, 1.0]]})

        summary = {
            "demo": "numpyforge-backend-mlops",
            "processed_dataset": str(processed_path),
            "artifact_dir": str(artifact.path),
            "model_version": artifact.metadata["version"],
            "test_accuracy": evaluation["classification"]["accuracy"],
            "api": {
                "health_status": health.status_code,
                "ready_status": ready.status_code,
                "metadata_status": metadata.status_code,
                "predict_status": prediction.status_code,
                "sample_prediction": prediction.json(),
            },
            "proof_points": [
                "pure NumPy model implementation",
                "versioned artifact loaded by API",
                "health/readiness separated",
                "CI covers quality, tests, pipeline smoke, and Docker build",
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


def _write_training_config(workspace: Path) -> Path:
    config = {
        "artifact": {
            "model_name": "numpyforge-portfolio-demo",
            "output_dir": str(workspace / "models" / "current"),
            "version": "portfolio-demo",
        },
        "data": {
            "processed_path": str(workspace / "data" / "binary_classification.npz"),
            "random_state": 42,
            "test_size": 0.25,
        },
        "mlflow": {
            "experiment_name": "numpyforge-portfolio-demo",
            "run_name": "portfolio-demo",
            "tags": {"demo": "portfolio"},
            "tracking_uri": f"sqlite:///{workspace / 'mlflow.db'}",
        },
        "model": {
            "batch_strategy": "batch",
            "early_stopping": True,
            "gradient_tol": 1e-6,
            "learning_rate": 0.2,
            "n_iter_no_change": 20,
            "n_iterations": 1000,
            "penalty": "l2",
            "random_state": 42,
            "regularization_strength": 0.01,
            "tol": 1e-6,
            "validation_fraction": 0.2,
        },
    }
    return _write_json(workspace / "training.json", config)


def _write_evaluation_config(workspace: Path) -> Path:
    config = {
        "artifact": {"artifact_dir": str(workspace / "models" / "current")},
        "data": {
            "processed_path": str(workspace / "data" / "binary_classification.npz"),
            "random_state": 42,
            "test_size": 0.25,
        },
    }
    return _write_json(workspace / "evaluation.json", config)


def _write_serving_config(workspace: Path) -> Path:
    config = {
        "artifact": {"artifact_dir": str(workspace / "models" / "current")},
        "service": {
            "name": "NumPyForge Portfolio Demo",
            "version": "0.1.0",
        },
    }
    return _write_json(workspace / "serving.json", config)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


@contextmanager
def _demo_serving_environment(serving_config: Path) -> Iterator[None]:
    env_vars = ("NUMPYFORGE_SERVING_CONFIG", "NUMPYFORGE_ARTIFACT_DIR")
    previous = {name: os.environ.get(name) for name in env_vars}
    os.environ["NUMPYFORGE_SERVING_CONFIG"] = str(serving_config)
    os.environ.pop("NUMPYFORGE_ARTIFACT_DIR", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
