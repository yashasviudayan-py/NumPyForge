# NumPyForge Case Study

## Problem

Most portfolio ML projects stop at a notebook. NumPyForge was built to show the full engineering arc:
the math behind the model, the test strategy that proves it, and the production path that serves it.

## What I Built

NumPyForge is a from-scratch machine-learning framework implemented with NumPy, wrapped in a
production-style MLOps loop. It includes classical models, neural networks, validation utilities,
evaluation reports, model artifacts, FastAPI serving, Docker packaging, and GitHub Actions CI.

## Technical Decisions

- **Pure NumPy model logic:** keeps gradient descent, regularization, backpropagation, and metrics
  inspectable instead of hiding them behind high-level ML libraries.
- **Scikit-like estimator API:** `fit`, `predict`, `predict_proba`, `score`, and fitted attributes
  make models easy to compose and test.
- **Artifact-first serving:** the API loads a saved artifact at startup and reports readiness
  separately from health. It does not silently train a fallback model on import.
- **JSON-ready reports:** evaluation output is suitable for CLIs, APIs, dashboards, or future model
  registry integrations.
- **CI as a release gate:** formatting, linting, typing, coverage, pipeline smoke tests, and Docker
  build smoke tests run on every push and pull request.

## Reliability Story

The project has unit tests for math helpers, validation, estimators, metrics, model selection,
artifacts, pipeline commands, and API behavior. Neural-network and logistic-regression gradients are
checked with finite differences so numerical correctness is tested at the level where bugs are most
expensive.

## Production Story

The Phase 5 pipeline creates deterministic demo data, trains a binary logistic classifier, tracks
metrics with MLflow when available, writes a versioned model artifact, and serves that artifact
through FastAPI. The serving app exposes `/health`, `/ready`, `/metadata`, and `/predict`, with
structured logs and clear failure modes for missing or corrupt artifacts.

## Tradeoffs

- The project optimizes for educational clarity over highly optimized kernels.
- The first release focuses on supervised learning and a binary production-serving path.
- Runtime artifacts are local and file-based; a richer registry, hosted deployment, and real
  datasets are intentionally left for future releases.

## Interview Sound Bite

> I built NumPyForge to show that I can reason from ML fundamentals all the way to production
> engineering: I implemented the models and gradients from scratch, then added the pieces that make
> them reliable and operable: tests, metrics, artifacts, FastAPI serving, Docker, and CI.
