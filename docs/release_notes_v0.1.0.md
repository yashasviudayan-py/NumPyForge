# NumPyForge v0.1.0

NumPyForge v0.1.0 is the first complete release of the project: a from-scratch NumPy machine
learning framework carried into a production-style Backend/MLOps workflow.

## Highlights

- Pure NumPy estimator architecture with validation, stable math helpers, deterministic randomness,
  and parameter serialization.
- Classical ML from scratch: linear regression, logistic regression, softmax regression, gradient
  descent variants, L1/L2 regularization, sample weights, and class weights.
- Neural-network layer from scratch: dense layers, activations, dropout, losses, initializers,
  backpropagation, SGD, momentum, RMSProp, Adam, learning-rate schedules, and MLP estimators.
- Evaluation toolkit: train/test splits, K-fold CV, stratified CV, grid/random search, baselines,
  classification/regression metrics, ROC/PR curves, and JSON-ready reports.
- Production loop: deterministic ingestion, MLflow-compatible tracking, versioned artifacts,
  FastAPI serving, structured logs, Docker, and Compose.
- CI/CD: Black, Ruff, mypy, pytest coverage, pipeline smoke tests, Docker build smoke tests, and
  pre-commit hooks.
- Portfolio demo kit: guided walkthrough, case study, API examples, architecture artwork, resume
  notes, and recording guide.

## Demo

```bash
python -m pip install -e ".[dev]"
make demo
```

The demo trains a model, writes a versioned artifact, loads it through FastAPI's app path, and
prints a compact JSON summary with accuracy, readiness, metadata, and sample prediction output.
