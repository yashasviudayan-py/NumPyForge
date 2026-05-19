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
