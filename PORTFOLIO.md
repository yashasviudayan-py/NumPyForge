# Portfolio Notes

Use this file to translate NumPyForge into resume bullets and interview talking points for
Backend/MLOps roles.

## Resume Bullets

Two-line version:

- Built NumPyForge, a from-scratch ML framework and production pipeline implementing linear
  regression, logistic/softmax regression, MLPs, optimizers, metrics, cross-validation, and
  finite-difference gradient checks with NumPy.
- Added an MLOps layer with deterministic ingestion, versioned model artifacts, MLflow-compatible
  tracking, FastAPI serving, Docker packaging, structured readiness/prediction logs, and GitHub
  Actions CI.

One-line version:

- Built a pure-NumPy ML framework with classical models, MLPs, evaluation tools, FastAPI serving,
  Docker packaging, versioned artifacts, and CI-enforced tests/type checks.

Compact version:

- NumPy ML framework + MLOps pipeline: from-scratch models, FastAPI serving, Docker, CI, 80 tests,
  and artifact-based deployment.

## Interview Talking Points

- Health vs readiness: `/health` confirms the process is alive; `/ready` confirms a valid model
  artifact is loaded. Prediction returns 503 when the artifact is unavailable.
- Artifact design: model weights, metadata, and metrics are stored together so training,
  evaluation, and serving share one contract.
- Testing strategy: unit tests cover math, validation, models, optimizers, metrics, API, pipeline,
  artifacts, deterministic behavior, and finite-difference gradient correctness.
- CI strategy: GitHub Actions runs format checks, linting, mypy, coverage reporting, pipeline
  smoke tests, and Docker build validation.
- Production tradeoff: model logic stays pure NumPy for educational transparency, while FastAPI,
  Docker, MLflow-compatible tracking, and CI provide the production shell.

## Built From Scratch

- Estimator base classes and fitted-state behavior.
- Vectorized numerical helpers including sigmoid, softmax, log-sum-exp, one-hot encoding, and
  probability clipping.
- Linear regression, binary logistic regression, multiclass softmax regression, and regularized
  gradient-descent training.
- Neural-network layers, activations, losses, initializers, MLP estimators, backpropagation, and
  optimizers.
- Train/test split, K-fold cross-validation, model search, metrics, reports, and baselines.

## Integrated For Production

- FastAPI for serving.
- Pydantic request/response validation.
- MLflow-compatible experiment tracking.
- Docker and Docker Compose for packaging.
- GitHub Actions, Black, Ruff, mypy, pytest, coverage, and pre-commit for quality enforcement.

## Strong Demo Script

Run:

```bash
make demo
```

Then explain:

1. Data is generated deterministically.
2. Training writes a versioned artifact.
3. Evaluation reports holdout metrics.
4. The FastAPI app loads the artifact.
5. The demo calls the API through `TestClient`, proving serving behavior without external infra.

## Repo Presentation Assets

- [Case study](CASE_STUDY.md) for a concise project narrative.
- [Demo walkthrough](DEMO.md) for recruiter/interviewer review.
- [API examples](docs/api_examples.http) for manual endpoint checks.
- [Demo recording guide](docs/demo_recording.md) for a short GIF or screen recording.
- [GitHub polish checklist](docs/github_polish.md) for repo topics, social preview, and pinned
  repo text.
