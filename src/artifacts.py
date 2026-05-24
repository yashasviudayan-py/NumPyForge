"""Versioned model artifact helpers for production-style workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.linear_model import LogisticRegression
from src.types import PathLikeString

ARTIFACT_SCHEMA_VERSION = 1
MODEL_TYPE = "binary_logistic_regression"
MODEL_FILENAME = "model.npz"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class ModelArtifact:
    """Loaded binary logistic model artifact."""

    model: LogisticRegression
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class ArtifactStatus:
    """Readiness status for a model artifact directory."""

    ready: bool
    path: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


def save_logistic_artifact(
    model: LogisticRegression,
    artifact_dir: PathLikeString,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> ModelArtifact:
    """Persist a fitted binary logistic model plus JSON metadata and metrics."""
    model._check_is_fitted()
    if model.classes_ is None or model.classes_.shape[0] != 2:
        raise ValueError("Only fitted binary LogisticRegression artifacts are supported.")

    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)

    artifact_metadata = _metadata_with_defaults(model, metadata)
    model.save_parameters(path / MODEL_FILENAME)
    _write_json(path / METADATA_FILENAME, artifact_metadata)
    _write_json(path / METRICS_FILENAME, metrics)

    return ModelArtifact(
        model=model,
        metadata=artifact_metadata,
        metrics=metrics,
        path=path,
    )


def load_logistic_artifact(artifact_dir: PathLikeString) -> ModelArtifact:
    """Load a versioned binary logistic model artifact from disk."""
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Artifact path is not a directory: {path}")

    metadata = _read_json(path / METADATA_FILENAME)
    metrics = _read_json(path / METRICS_FILENAME)
    _validate_metadata(metadata)

    hyperparameters = cast(dict[str, Any], metadata.get("hyperparameters", {}))
    model = LogisticRegression(**_supported_logistic_hyperparameters(hyperparameters))
    model.load_parameters(path / MODEL_FILENAME)

    if model.classes_ is None or model.classes_.shape[0] != 2:
        raise ValueError("Loaded artifact is not a binary LogisticRegression model.")

    return ModelArtifact(model=model, metadata=metadata, metrics=metrics, path=path)


def artifact_status(artifact_dir: PathLikeString) -> ArtifactStatus:
    """Return readiness information without raising load errors."""
    path = Path(artifact_dir)
    try:
        artifact = load_logistic_artifact(path)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return ArtifactStatus(ready=False, path=str(path), error=str(exc))

    return ArtifactStatus(
        ready=True,
        path=str(path),
        metadata=artifact.metadata,
    )


def _metadata_with_defaults(model: LogisticRegression, metadata: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(UTC).isoformat()
    default_metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "model_name": "numpyforge-logistic-regression",
        "version": created_at,
        "created_at": created_at,
        "n_features": model.n_features_in_,
        "classes": [] if model.classes_ is None else model.classes_.astype(int).tolist(),
        "hyperparameters": _logistic_hyperparameters(model),
    }
    default_metadata.update(metadata)
    default_metadata.update(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "n_features": model.n_features_in_,
            "classes": [] if model.classes_ is None else model.classes_.astype(int).tolist(),
            "hyperparameters": _logistic_hyperparameters(model),
        }
    )
    return default_metadata


def _logistic_hyperparameters(model: LogisticRegression) -> dict[str, Any]:
    return {
        "learning_rate": model.learning_rate,
        "n_iterations": model.n_iterations,
        "fit_intercept": model.fit_intercept,
        "penalty": model.penalty,
        "regularization_strength": model.regularization_strength,
        "threshold": model.threshold,
        "multi_class": model.multi_class,
        "batch_strategy": model.batch_strategy,
        "batch_size": model.batch_size,
        "shuffle": model.shuffle,
        "tol": model.tol,
        "gradient_tol": model.gradient_tol,
        "early_stopping": model.early_stopping,
        "n_iter_no_change": model.n_iter_no_change,
        "validation_fraction": model.validation_fraction,
        "class_weight": model.class_weight,
        "random_state": model.random_state if isinstance(model.random_state, int) else None,
    }


def _supported_logistic_hyperparameters(values: dict[str, Any]) -> dict[str, Any]:
    supported_names = set(_logistic_hyperparameters(LogisticRegression()))
    return {name: value for name, value in values.items() if name in supported_names}


def _validate_metadata(metadata: dict[str, Any]) -> None:
    schema_version = metadata.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema version: {schema_version}. "
            f"Expected {ARTIFACT_SCHEMA_VERSION}."
        )
    model_type = metadata.get("model_type")
    if model_type != MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected {MODEL_TYPE}.")
    if "hyperparameters" not in metadata:
        raise ValueError("Artifact metadata is missing 'hyperparameters'.")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
