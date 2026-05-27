.PHONY: ingest train evaluate demo serve quality test coverage docker-build compose-up

ingest:
	python -m pipeline.ingest

train:
	python -m pipeline.train

evaluate:
	python -m pipeline.evaluate

demo:
	python examples/portfolio_demo.py

serve:
	uvicorn api.main:app --reload

quality:
	python -m black --check .
	python -m ruff check .
	python -m mypy src api pipeline tests

test:
	python -m pytest

coverage:
	python -m pytest --cov=src --cov=api --cov=pipeline --cov-report=term-missing

docker-build:
	docker build -t numpyforge:latest .

compose-up:
	docker compose up --build api
