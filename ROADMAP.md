# Roadmap

NumPyForge v0.1.0 completes the first full arc: from-scratch models, validation, MLOps, serving,
and CI. Future work should expand capability without weakening the project's learning-first,
NumPy-first character.

## Candidate v0.2.0 Directions

- Add more CS229-style algorithms: SVMs, PCA, k-means, Gaussian discriminant analysis, decision
  trees, and random forests.
- Expand model artifact support beyond binary logistic regression to MLP classifiers/regressors
  and linear regression.
- Add richer model registry behavior: multiple artifact versions, active-model switching, and
  rollback metadata.
- Support real dataset ingestion through CSV/NPZ configuration while keeping deterministic demo
  data available for tests.
- Build a documentation site with equations, implementation notes, API examples, and end-to-end
  tutorials.
- Add coverage thresholds and optional Docker image publishing once release workflows stabilize.
