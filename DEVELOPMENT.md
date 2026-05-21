# Development Roadmap: From-Scratch ML Framework & Production Pipeline

This roadmap turns NumPyForge into a learning-first machine learning framework that connects CS229 theory with production-oriented MLOps practice. Each phase should produce working code, tests, documentation, and examples that make the mathematical ideas inspectable.

## Guiding Principles

- Implement core algorithms from first principles using NumPy before reaching for higher-level libraries.
- Keep estimator APIs familiar: `fit`, `predict`, `predict_proba`, `score`, and typed configuration.
- Prefer vectorized operations and explicit matrix shapes over hidden loops.
- Treat numerical correctness as a product requirement, not an afterthought.
- Make every model trainable locally, testable in CI, trackable in MLflow, and servable through FastAPI.

## Phase 1: Core Math Engine & Base Architecture

Build the shared foundation for estimators, tensor-like NumPy utilities, validation, and optimization-ready abstractions.

### CS229 Concepts Applied

- Linear algebra foundations: vectors, matrices, dot products, norms, broadcasting.
- Matrix calculus basics: gradients, Jacobians, Hessian-aware interfaces.
- Convex optimization setup: objective functions, gradients, convergence criteria.
- Bias-variance framing for reusable estimator design.

### Tasks

- [x] Define a stable `BaseEstimator` interface with `fit`, `predict`, `score`, and fitted-state checks.
- [x] Split supervised model abstractions into classifier and regressor base classes.
- [x] Add reusable validation helpers for feature matrices, target vectors, class labels, and sample weights.
- [x] Standardize NumPy type aliases for float arrays, integer arrays, boolean masks, and model parameters.
- [x] Implement vectorized math utilities for sigmoid, softmax, log-sum-exp, one-hot encoding, norms, and clipping.
- [x] Add deterministic random-state handling across models, initializers, train/test splits, and optimizers.
- [x] Create reusable loss-function protocols for objective value and gradient computation.
- [x] Add parameter serialization helpers for saving and loading fitted model weights.
- [x] Write developer examples showing the expected estimator lifecycle.
- [x] Document shape conventions for `(n_samples, n_features)`, `(n_samples,)`, and `(n_classes,)`.

### Deliverables

- `src/base.py` with production-quality estimator abstractions.
- `src/math.py` or `src/utils/math.py` with vectorized numerical helpers.
- `src/validation.py` for shared input validation.
- Typed docstrings explaining array shapes and return values.
- Unit tests for validation behavior, numerical helpers, and fitted-state checks.

## Phase 2: Classical ML Models from Scratch

Implement classical supervised learning models using pure NumPy and optimization logic that exposes the underlying CS229 math.

### CS229 Concepts Applied

- Logistic regression as maximum likelihood estimation.
- Gradient descent and stochastic gradient descent.
- L1 and L2 regularization as constrained optimization and MAP estimation.
- Newton's Method and Hessian intuition for convex objectives.
- Decision boundaries, margins, likelihoods, and loss surfaces.

### Tasks

- [x] Expand binary logistic regression with stable loss computation and probability calibration.
- [x] Add multiclass logistic regression using softmax and cross-entropy.
- [x] Implement linear regression with closed-form normal equation support.
- [x] Implement gradient-descent linear regression for large datasets.
- [x] Add L2 regularization for linear and logistic regression.
- [x] Add L1 regularization with subgradient or proximal update support.
- [x] Implement configurable batch, mini-batch, and stochastic gradient descent.
- [x] Add early stopping based on validation loss and gradient norm.
- [x] Track loss history, parameter norms, convergence status, and number of iterations.
- [x] Add optional class weights and sample weights for imbalanced datasets.
- [x] Build examples that compare learned coefficients with known synthetic data.
- [x] Document where the implementation matches or intentionally differs from scikit-learn.

### Deliverables

- `src/linear_model.py` with linear regression, binary logistic regression, and multiclass logistic regression.
- `src/optimizers.py` with reusable gradient-descent variants.
- `examples/classical_ml.ipynb` or script-based examples under `examples/`.
- Tests for convergence, regularization effects, shape validation, and deterministic training.
- README section explaining the CS229 objective functions behind each model.

## Phase 3: Deep Learning Layer

Build a small neural-network module that teaches the mechanics of forward passes, backpropagation, initialization, and optimizers without relying on PyTorch or TensorFlow.

### CS229 Concepts Applied

- Backpropagation through computational graphs.
- Matrix calculus and chain rule.
- Universal approximation intuition for multilayer perceptrons.
- Nonlinear activation functions and vanishing gradients.
- Optimization dynamics: SGD, momentum, RMSProp-style scaling, and Adam.

### Tasks

- [ ] Define a `Layer` protocol with `forward`, `backward`, `params`, and `grads`.
- [ ] Implement dense fully connected layers using pure NumPy.
- [ ] Implement activation functions: ReLU, Leaky ReLU, sigmoid, tanh, and softmax.
- [ ] Implement loss functions for binary cross-entropy, categorical cross-entropy, and mean squared error.
- [ ] Add parameter initializers: zeros, random normal, Xavier/Glorot, and He initialization.
- [ ] Implement a sequential `MLPClassifier` and `MLPRegressor`.
- [ ] Implement backpropagation with cached intermediate activations.
- [ ] Add optimizers: SGD, SGD with momentum, RMSProp, and Adam.
- [ ] Support mini-batch training with deterministic shuffling.
- [ ] Add dropout as an optional regularization layer.
- [ ] Add learning-rate schedules for constant, step decay, and cosine decay.
- [ ] Add training-history output for loss, metrics, learning rate, and epoch timing.
- [ ] Create toy demos for XOR, moons, blobs, and simple regression curves.

### Deliverables

- `src/neural_network/` package with layers, activations, losses, initializers, and MLP estimators.
- Pure NumPy optimizers with tests against hand-computed updates.
- Gradient-checking utilities for validating backpropagation.
- Example notebooks or scripts visualizing training curves and decision boundaries.
- Documentation explaining the forward and backward equations for each layer.

## Phase 4: Validation & Evaluation Pipeline

Create a framework-native model-selection and evaluation layer so experiments can be trusted before they are served.

### CS229 Concepts Applied

- Generalization error, empirical risk, and validation risk.
- Bias-variance tradeoff.
- Classification thresholds, precision-recall tradeoffs, and ROC analysis.
- Cross-validation for model selection.
- Statistical uncertainty and confidence intervals for metrics.

### Tasks

- [ ] Implement deterministic train/test split with optional stratification.
- [ ] Implement K-fold and stratified K-fold cross-validation from scratch.
- [ ] Add classification metrics: accuracy, precision, recall, F1-score, confusion matrix, log loss, ROC-AUC, and PR-AUC.
- [ ] Add regression metrics: MAE, MSE, RMSE, R2, adjusted R2, and explained variance.
- [ ] Implement ROC and precision-recall curve generation.
- [ ] Add threshold-tuning utilities for binary classifiers.
- [ ] Add model-selection helpers for grid search and randomized search.
- [ ] Add baseline estimators such as majority-class classifier and mean regressor.
- [ ] Create evaluation reports that return structured dictionaries and optional plots.
- [ ] Add metric tests using small examples with hand-verified expected values.
- [ ] Add integration tests comparing selected metrics against scikit-learn on tiny datasets where appropriate.

### Deliverables

- `src/model_selection.py` for splits, folds, and search utilities.
- `src/metrics.py` for classification and regression metrics.
- `pipeline/evaluate.py` for repeatable local evaluation runs.
- Evaluation artifacts such as JSON reports and optional plots.
- Documentation showing how to choose metrics for imbalanced classification and regression.

## Phase 5: Production & MLOps Infrastructure

Turn the learning framework into a practical training and serving system with repeatable experiments, model artifacts, and deployment-ready APIs.

### CS229 Concepts Applied

- Train/validation/test discipline as protection against overfitting.
- Model selection under operational constraints.
- Calibration, monitoring, and distribution shift awareness.
- Reproducibility of empirical results.

### Tasks

- [ ] Expand FastAPI serving with health, readiness, metadata, and prediction endpoints.
- [ ] Add request and response schemas with strict validation and helpful error messages.
- [ ] Implement model artifact loading from `models/` with versioned metadata.
- [ ] Add a training pipeline that logs parameters, metrics, plots, and artifacts to MLflow.
- [ ] Add MLflow experiment naming, run tags, and reproducibility metadata.
- [ ] Add data-ingestion entrypoints for raw and processed datasets.
- [ ] Create configuration files for training, evaluation, and serving.
- [ ] Add Dockerfile improvements for reproducible runtime builds.
- [ ] Add a Docker Compose setup for API plus optional MLflow tracking server.
- [ ] Add structured logging for training and serving.
- [ ] Add basic prediction monitoring hooks for latency, input shape, and output distribution.
- [ ] Add Makefile or task-runner commands for common workflows.
- [ ] Document local development, training, serving, and artifact-management flows.

### Deliverables

- Production-ready FastAPI app under `api/`.
- `pipeline/train.py`, `pipeline/evaluate.py`, and `pipeline/ingest.py` entrypoints.
- MLflow-backed experiment tracking with persisted run metadata.
- Docker image that can train or serve the project.
- Local runbook for training a model, registering an artifact, and serving predictions.

## Phase 6: Testing & CI/CD

Make correctness enforceable with tests, numerical checks, linting, type checking, and GitHub Actions.

### CS229 Concepts Applied

- Numerical gradient checking for validating analytic gradients.
- Finite differences and approximation error.
- Statistical test design for stochastic algorithms.
- Reproducibility through deterministic seeds and controlled randomness.

### Tasks

- [ ] Add pytest unit tests for every estimator, metric, optimizer, and validation utility.
- [ ] Add finite-difference gradient checks for logistic regression, MLP layers, and loss functions.
- [ ] Add tests for deterministic behavior under fixed random seeds.
- [ ] Add convergence tests on small synthetic datasets.
- [ ] Add edge-case tests for empty arrays, wrong shapes, invalid labels, NaNs, and infinities.
- [ ] Add smoke tests for FastAPI health and prediction endpoints.
- [ ] Add Docker build smoke tests.
- [ ] Configure GitHub Actions for pytest, Black, Ruff, and mypy.
- [ ] Add coverage reporting and minimum coverage thresholds once the codebase stabilizes.
- [ ] Add pre-commit configuration for local formatting and linting.
- [ ] Add CI caching for Python dependencies.
- [ ] Add release checklist for tags, changelog updates, and version bumps.

### Deliverables

- Comprehensive `tests/` coverage across math, models, metrics, pipeline, and API.
- `tests/test_gradients.py` with finite-difference gradient checks.
- `.github/workflows/ci.yml` running format, lint, type, and test checks.
- Optional `.pre-commit-config.yaml` for local quality gates.
- CI badge in `README.md` after the public GitHub workflow is active.

## Suggested Milestones

### Milestone 1: Framework Skeleton

- Complete Phase 1.
- Keep all current tests passing.
- Publish documentation for estimator conventions and shape contracts.

### Milestone 2: Classical ML MVP

- Complete the core items in Phase 2.
- Train logistic and linear regression on synthetic datasets.
- Add convergence and regularization tests.

### Milestone 3: Evaluation-Ready Framework

- Complete Phase 4 metrics and validation splits.
- Add model-selection utilities.
- Produce repeatable evaluation reports.

### Milestone 4: NumPy Deep Learning MVP

- Complete dense layers, activations, losses, backpropagation, and optimizers from Phase 3.
- Validate gradients through finite-difference checks.
- Demonstrate XOR and multiclass toy classification.

### Milestone 5: Production Loop

- Complete Phase 5.
- Train, track, save, load, and serve a model artifact.
- Package the API in Docker.

### Milestone 6: CI-Enforced Quality

- Complete Phase 6.
- Require automated checks before merging.
- Maintain documented commands for local parity with CI.

## Definition of Done

Each feature is considered complete only when it has:

- Working implementation with typed public interfaces.
- Unit tests covering normal behavior and important edge cases.
- Numerical validation where gradients or optimization are involved.
- Documentation explaining the CS229 concept and implementation choices.
- Reproducible example or pipeline entrypoint when user-facing.
- Passing Black, Ruff, mypy, and pytest checks.

## Local Quality Commands

Run these before committing:

```bash
python -m black --check .
python -m ruff check .
python -m mypy src api pipeline tests
python -m pytest
```

To format and lint locally:

```bash
python -m black .
python -m ruff check . --fix
```
