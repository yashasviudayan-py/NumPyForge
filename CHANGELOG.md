# Changelog

## Unreleased

### Changed

- Renamed the core library package from `src` to `numpyforge`, so it is imported as
  `from numpyforge... import ...` to match the distribution name. All internal modules, tests,
  examples, pipeline/API entrypoints, tooling config (ruff, coverage, mypy, packaging), CI, and
  docs were updated accordingly. Added a `py.typed` marker so the package ships as typed.

### Fixed

- `roc_auc_score`/`pr_auc_score` use `np.trapezoid`, which only exists in NumPy 2.0+. The dependency
  floor was raised to `numpy>=2.0.0` (in `pyproject.toml` and `requirements.txt`) so the declared
  range matches the code and the metrics no longer break on NumPy 1.26.x.

### Added

- MIT `LICENSE` file plus `license` metadata and trove classifiers in `pyproject.toml`.
- Coverage threshold enforcement (`fail_under = 85`) so CI fails on coverage regressions.
- Optional scikit-learn parity tests (`tests/test_sklearn_parity.py`, `[test]` extra) that
  cross-check the from-scratch linear models and metrics against scikit-learn within machine-epsilon
  tolerances. The suite is skipped when scikit-learn is absent.

## v0.1.0 - 2026-05-29

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
- Portfolio demo kit with a guided walkthrough, case study, API request examples, architecture
  artwork, resume/interview notes, and a short demo recording guide.

### Notes

- Model logic is implemented with NumPy and the Python standard library where practical.
- Coverage is reported but not threshold-enforced in this release.
- Docker images are built in CI for smoke validation but are not published.
