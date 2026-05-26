# Changelog

## v0.1.0 - 2026-05-26

Initial release of NumPyForge as a from-scratch ML framework and production pipeline.

### Added

- Core estimator architecture with validation, deterministic randomness, math helpers, and
  parameter serialization.
- Classical ML models: linear regression, binary logistic regression, multiclass softmax
  regression, gradient descent variants, L1/L2 regularization, sample weights, and class weights.
- Pure NumPy neural-network layer with dense layers, activations, dropout, losses, initializers,
  SGD, momentum, RMSProp, Adam, learning-rate schedules, and MLP classifier/regressor estimators.
- Validation and evaluation utilities including train/test splits, K-fold and stratified K-fold
  cross-validation, grid/randomized search, baselines, classification metrics, regression metrics,
  ROC/PR curves, and JSON-ready reports.
- Production workflow with deterministic ingestion, training, evaluation, versioned artifacts,
  local MLflow tracking support, FastAPI serving, structured logs, Docker, and Docker Compose.
- CI/CD tooling with GitHub Actions, Black, Ruff, mypy, pytest coverage reporting, pipeline smoke
  tests, Docker build smoke tests, and pre-commit hooks.

### Notes

- Model logic is implemented with NumPy and the Python standard library where practical.
- Coverage is reported but not threshold-enforced in this release.
- Docker images are built in CI for smoke validation but are not published.
