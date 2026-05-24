.PHONY: ingest train evaluate serve quality test docker-build compose-up

ingest:
	python -m pipeline.ingest

train:
	python -m pipeline.train

evaluate:
	python -m pipeline.evaluate

serve:
	uvicorn api.main:app --reload

quality:
	python -m black --check .
	python -m ruff check .
	python -m mypy src api pipeline tests

test:
	python -m pytest

docker-build:
	docker build -t numpyforge:latest .

compose-up:
	docker compose up --build api
