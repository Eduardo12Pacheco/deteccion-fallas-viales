UV ?= uv
PYTHON ?= $(UV) run python

.DEFAULT_GOAL := help

.PHONY: help setup lint types test test-full fixtures data train eval serve build ci-local

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the locked environment
	$(UV) sync

lint: ## Run the linter and the formatter check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

types: ## Run the strict type checker
	$(UV) run mypy --strict src

test: ## Run the fast test suite
	$(UV) run pytest -q

test-full: ## Run the full suite with the coverage gate
	$(UV) run pytest -q --cov=src/rdv --cov-report=term-missing

fixtures: ## Regenerate the committed test fixtures
	$(PYTHON) scripts/make_fixtures.py

data: ## Download the RDD2022 countries used for training
	$(UV) run rdv data download --countries Japan India Czech

train: ## Fine-tune the detector from the pinned configuration
	$(UV) run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml

eval: ## Evaluate the exported model on the held-out test split
	$(UV) run rdv evaluate --data "$${RDV_DATA_DIR}/rdd2022" --splits "$${RDV_DATA_DIR}/rdd2022/splits.json" --split test --out reports/v1

serve: ## Start the local web app
	$(UV) run rdv serve

build: ## Build the wheel and the source distribution
	$(UV) build

ci-local: lint types test ## Run the continuous-integration checks locally
