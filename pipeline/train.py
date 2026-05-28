"""Train and register the Phase 5 binary logistic model artifact."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import numpy as np

try:
    _mlflow: Any = importlib.import_module("mlflow")
except ModuleNotFoundError:
    _mlflow = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest import ingest  # noqa: E402
from src.artifacts import METADATA_FILENAME, METRICS_FILENAME, save_logistic_artifact  # noqa: E402
from src.config import Config, get_config_value, load_json_config  # noqa: E402
from src.linear_model import LogisticRegression  # noqa: E402
from src.metrics import classification_report_dict, log_loss  # noqa: E402
from src.model_selection import train_test_split  # noqa: E402
from src.types import FloatArray, IntArray  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.json"
mlflow: Any = _mlflow if _mlflow is not None else None


def train(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Train, evaluate, track, and save a binary logistic model artifact."""
    config = load_json_config(config_path)
    processed_path = _ensure_dataset(config, config_path)
    features, targets = _load_dataset(processed_path)
    test_size = get_config_value(config, "data.test_size", float)
    random_state = get_config_value(config, "data.random_state", int)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )

    model = LogisticRegression(**_model_params(config))
    tracker = _mlflow_client()
    _configure_mlflow(config)
    experiment_name = get_config_value(config, "mlflow.experiment_name", str)
    run_name = get_config_value(config, "mlflow.run_name", str)
    tracker.set_experiment(experiment_name)

    with tracker.start_run(run_name=run_name) as run:
        tags = cast(dict[str, str], config.get("mlflow", {}).get("tags", {}))
        if tags:
            tracker.set_tags(tags)
        tracker.log_params(_model_params(config))
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        predictions = model.predict(X_test)
        metrics = classification_report_dict(y_test, predictions, probabilities[:, 1])
        metrics["log_loss"] = log_loss(y_test, probabilities)
        metrics["train_accuracy"] = model.score(X_train, y_train)
        metrics["test_accuracy"] = model.score(X_test, y_test)
        metrics["n_train_samples"] = int(X_train.shape[0])
        metrics["n_test_samples"] = int(X_test.shape[0])
        _log_flat_metrics(tracker, metrics)

        artifact_dir = PROJECT_ROOT / get_config_value(config, "artifact.output_dir", str)
        metadata = {
            "model_name": get_config_value(config, "artifact.model_name", str),
            "version": get_config_value(config, "artifact.version", str),
            "training": {
                "config_path": str(config_path),
                "processed_path": str(processed_path),
                "random_state": random_state,
                "test_size": test_size,
                "mlflow_run_id": run.info.run_id,
            },
        }
        artifact = save_logistic_artifact(
            model,
            artifact_dir,
            metadata=metadata,
            metrics=metrics,
        )
        tracker.log_artifact(str(artifact.path / METADATA_FILENAME), artifact_path="model")
        tracker.log_artifact(str(artifact.path / METRICS_FILENAME), artifact_path="model")
        tracker.log_artifact(str(artifact.path / "model.npz"), artifact_path="model")

    return {
        "artifact_dir": str(artifact.path),
        "metrics": metrics,
        "metadata": artifact.metadata,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run training and print a compact JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.config), indent=2, sort_keys=True))


def _ensure_dataset(config: Config, config_path: Path) -> Path:
    processed_path = PROJECT_ROOT / get_config_value(config, "data.processed_path", str)
    if not processed_path.exists():
        return ingest(config_path)
    return processed_path


def _load_dataset(path: Path) -> tuple[FloatArray, IntArray]:
    with np.load(path, allow_pickle=False) as dataset:
        features = cast(FloatArray, dataset["X"].astype(np.float64))
        targets = cast(IntArray, dataset["y"].astype(np.int_))
    return features, targets


def _model_params(config: Config) -> dict[str, Any]:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise TypeError("Config value 'model' must be an object.")
    return dict(model_config)


def _configure_mlflow(config: Config) -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        get_config_value(config, "mlflow.tracking_uri", str),
    )
    if tracking_uri.startswith("file:") and not tracking_uri.startswith("file://"):
        relative_path = tracking_uri.removeprefix("file:")
        tracking_uri = (PROJECT_ROOT / relative_path).resolve().as_uri()
    _mlflow_client().set_tracking_uri(tracking_uri)


def _log_flat_metrics(tracker: Any, metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, int | float):
            tracker.log_metric(name, float(value))


def _mlflow_client() -> Any:
    global mlflow
    if mlflow is None:
        mlflow = _FallbackMlflow()
    return mlflow


class _FallbackRunInfo:
    def __init__(self) -> None:
        self.run_id = "mlflow-unavailable"


class _FallbackRun:
    def __init__(self) -> None:
        self.info = _FallbackRunInfo()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class _FallbackMlflow:
    def set_tracking_uri(self, tracking_uri: str) -> None:
        if tracking_uri.startswith("file://"):
            Path(tracking_uri.removeprefix("file://")).mkdir(parents=True, exist_ok=True)

    def set_experiment(self, experiment_name: str) -> None:
        _ = experiment_name

    def start_run(self, *, run_name: str) -> _FallbackRun:
        _ = run_name
        return _FallbackRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        _ = tags

    def log_params(self, params: dict[str, Any]) -> None:
        _ = params

    def log_metric(self, name: str, value: float) -> None:
        _ = name
        _ = value

    def log_artifact(self, local_path: str, *, artifact_path: str) -> None:
        _ = local_path
        _ = artifact_path


if __name__ == "__main__":
    main()
