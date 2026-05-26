# Release Checklist

Use this checklist to cut a NumPyForge release.

## v0.1.0

- [ ] Confirm `pyproject.toml` version is `0.1.0`.
- [ ] Run local quality checks:
  - `python -m black --check .`
  - `python -m ruff check .`
  - `python -m mypy src api pipeline tests`
  - `python -m pytest --cov=src --cov=api --cov=pipeline --cov-report=term-missing`
- [ ] Run local pipeline smoke checks:
  - `python -m pipeline.ingest`
  - `python -m pipeline.train`
  - `python -m pipeline.evaluate`
- [ ] Run Docker smoke build:
  - `docker build -t numpyforge:local .`
- [ ] Confirm GitHub Actions is green on `main`.
- [ ] Create and push the release tag:
  - `git tag v0.1.0`
  - `git push origin v0.1.0`
- [ ] Create the GitHub release using the `CHANGELOG.md` `v0.1.0` notes.

## Release Policy

- Keep runtime data, model artifacts, and MLflow runs out of git.
- Do not publish Docker images until an explicit distribution target is chosen.
- Add a coverage threshold only after the project has stabilized beyond `v0.1.0`.
