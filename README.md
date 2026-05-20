# NumPyForge

Custom machine learning components implemented from scratch with NumPy, plus a production-oriented training and serving scaffold.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
uvicorn api.main:app --reload
```

## Project Layout

```text
.
├── api/                  # FastAPI application for model serving
├── configs/              # Runtime and experiment configuration
├── data/                 # Local data, ignored by git
├── examples/             # Developer examples and learning walkthroughs
├── models/               # Serialized model artifacts, ignored by git
├── pipeline/             # Data ingestion and training entrypoints
├── src/                  # Core NumPy ML library
└── tests/                # Unit tests
```

## Development Goals

- Implement core ML algorithms from first principles using NumPy.
- Keep model APIs close to scikit-learn: `fit`, `predict`, and typed configuration.
- Track experiments and artifacts with MLflow.
- Serve trained models through FastAPI.
- Keep the project Docker-ready from the beginning.

## Framework Conventions

NumPyForge estimators follow a small, explicit contract inspired by scikit-learn:

- `fit(X, y)` learns parameters from a feature matrix `X` with shape `(n_samples, n_features)`
  and a target vector `y` with shape `(n_samples,)`.
- `predict(X)` returns predictions with shape `(n_samples,)`.
- `score(X, y)` returns the estimator's default quality score: accuracy for classifiers and R2
  for regressors.
- Fitted attributes end in `_`, such as `weights_`, `bias_`, and `n_features_in_`.
- `save_parameters(path)` and `load_parameters(path)` round-trip fitted model state through NumPy
  `.npz` archives.

Core Phase 1 utilities live in:

- `src/base.py` for estimator abstractions, classifier/regressor bases, fitted-state checks, loss
  protocols, and parameter serialization.
- `src/validation.py` for reusable feature, target, class-label, and sample-weight validation.
- `src/math.py` for stable vectorized operations such as sigmoid, softmax, log-sum-exp, one-hot
  encoding, clipping, and L2 norms.
- `src/random.py` for deterministic random-state handling.

Run the estimator lifecycle example with:

```bash
python examples/estimator_lifecycle.py
```
