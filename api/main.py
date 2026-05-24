"""FastAPI app for serving versioned NumPyForge model artifacts."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from src.artifacts import ModelArtifact, artifact_status, load_logistic_artifact
from src.config import get_config_value, load_json_config
from src.logging_utils import configure_json_logging, log_json
from src.types import FloatArray
from src.validation import check_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "serving.json"
CONFIG_ENV_VAR = "NUMPYFORGE_SERVING_CONFIG"
ARTIFACT_ENV_VAR = "NUMPYFORGE_ARTIFACT_DIR"

logger = logging.getLogger("numpyforge.api")


class PredictionRequest(PydanticBaseModel):
    """Prediction request containing a 2D feature matrix."""

    features: list[list[float]] = Field(min_length=1)


class PredictionResponse(PydanticBaseModel):
    """Prediction response for binary classification."""

    predictions: list[int]
    probabilities: list[float]
    model_version: str
    latency_ms: float


class HealthResponse(PydanticBaseModel):
    """Liveness response."""

    status: str


class ReadinessResponse(PydanticBaseModel):
    """Model readiness response."""

    ready: bool
    artifact_dir: str
    model_version: str | None = None
    error: str | None = None


class MetadataResponse(PydanticBaseModel):
    """Loaded model metadata response."""

    metadata: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class ServingState:
    """Mutable serving state owned by the FastAPI app."""

    artifact_dir: Path
    artifact: ModelArtifact | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.error is None


def create_app(config_path: Path | None = None) -> FastAPI:
    """Create the FastAPI application and load the configured model artifact."""
    configure_json_logging()
    resolved_config_path = config_path or Path(os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG))
    config = load_json_config(resolved_config_path)
    artifact_dir = Path(
        os.getenv(ARTIFACT_ENV_VAR, get_config_value(config, "artifact.artifact_dir", str))
    )
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir

    app = FastAPI(
        title=get_config_value(config, "service.name", str),
        version=get_config_value(config, "service.version", str),
    )
    state = _load_state(artifact_dir)
    app.state.serving_state = state

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        current_state = _state(app)
        response = _readiness_response(current_state)
        if not current_state.ready:
            log_json(
                logger,
                logging.WARNING,
                "readiness_failed",
                artifact_dir=str(current_state.artifact_dir),
                error=current_state.error,
            )
            raise HTTPException(status_code=503, detail=response.model_dump())
        return response

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        artifact = _require_artifact(app)
        return MetadataResponse(metadata=artifact.metadata, metrics=artifact.metrics)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        artifact = _require_artifact(app)
        started = time.perf_counter()
        try:
            features = cast(FloatArray, check_feature_matrix(np.asarray(request.features)))
            probabilities = artifact.model.predict_proba(features)
            predictions = artifact.model.predict(features)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        positive_probabilities = probabilities[:, 1]
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = PredictionResponse(
            predictions=[int(prediction) for prediction in predictions],
            probabilities=[float(probability) for probability in positive_probabilities],
            model_version=str(artifact.metadata.get("version", "unknown")),
            latency_ms=latency_ms,
        )
        log_json(
            logger,
            logging.INFO,
            "prediction_served",
            latency_ms=round(latency_ms, 6),
            n_features=int(features.shape[1]),
            n_predictions=int(features.shape[0]),
            probability_min=float(np.min(positive_probabilities)),
            probability_mean=float(np.mean(positive_probabilities)),
            probability_max=float(np.max(positive_probabilities)),
        )
        return response

    return app


def _load_state(artifact_dir: Path) -> ServingState:
    try:
        artifact = load_logistic_artifact(artifact_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        log_json(
            logger,
            logging.WARNING,
            "artifact_unavailable",
            artifact_dir=str(artifact_dir),
            error=str(exc),
        )
        return ServingState(artifact_dir=artifact_dir, error=str(exc))

    log_json(
        logger,
        logging.INFO,
        "artifact_loaded",
        artifact_dir=str(artifact_dir),
        model_version=str(artifact.metadata.get("version", "unknown")),
    )
    return ServingState(artifact_dir=artifact_dir, artifact=artifact)


def _state(app: FastAPI) -> ServingState:
    return cast(ServingState, app.state.serving_state)


def _readiness_response(state: ServingState) -> ReadinessResponse:
    if state.ready and state.artifact is not None:
        return ReadinessResponse(
            ready=True,
            artifact_dir=str(state.artifact_dir),
            model_version=str(state.artifact.metadata.get("version", "unknown")),
        )
    status = artifact_status(state.artifact_dir)
    return ReadinessResponse(
        ready=False,
        artifact_dir=str(state.artifact_dir),
        error=state.error or status.error,
    )


def _require_artifact(app: FastAPI) -> ModelArtifact:
    state = _state(app)
    if not state.ready or state.artifact is None:
        response = _readiness_response(state)
        raise HTTPException(status_code=503, detail=response.model_dump())
    return state.artifact


app = create_app()
