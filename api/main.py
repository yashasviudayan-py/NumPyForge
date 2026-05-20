"""FastAPI app for serving model predictions."""

from __future__ import annotations

from typing import cast

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel

from src.linear_model import LogisticRegression
from src.types import FloatArray


class PredictionRequest(PydanticBaseModel):
    features: list[list[float]]


class PredictionResponse(PydanticBaseModel):
    predictions: list[int]
    probabilities: list[float]


app = FastAPI(title="NumPy ML Serving API", version="0.1.0")

_model = LogisticRegression(learning_rate=0.1, n_iterations=1_000, random_state=42)
_model.fit(
    np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    np.array([0, 0, 0, 1], dtype=np.int_),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    features = cast(FloatArray, np.asarray(request.features, dtype=np.float64))
    probabilities = _model.predict_proba(features)
    predictions = _model.predict(features)
    return PredictionResponse(
        predictions=[int(prediction) for prediction in predictions],
        probabilities=[float(probability) for probability in probabilities],
    )
