# Convenience wrappers. Everything here also works as a plain command.
.PHONY: help install corpus seed run worker test eval tune docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create a venv and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

corpus:  ## Download the real CC-licensed corpus (needs network)
	.venv/bin/python -m scripts.fetch_corpus

corpus-offline:  ## Generate the offline placeholder corpus (no network)
	.venv/bin/python -m scripts.make_placeholder_corpus

seed:  ## Migrate, ingest, load posts and run the whole pipeline
	.venv/bin/python -m scripts.seed

run:  ## Start the API (worker runs in-process)
	.venv/bin/uvicorn app.main:app --reload --port 8000

worker:  ## Start a standalone worker
	.venv/bin/python -m app.worker

test:  ## Run the test suite
	.venv/bin/python -m pytest

eval:  ## Score the labeled eval set (PROBE 5)
	.venv/bin/python -m scripts.run_eval --json eval/report.json

tune:  ## Sweep the similarity threshold against the eval set
	.venv/bin/python -m scripts.tune_thresholds

docker:  ## Bring the whole stack up on Postgres
	docker compose up --build

clean:
	rm -f flyrank.sqlite3 && rm -rf .pytest_cache eval/report.json
