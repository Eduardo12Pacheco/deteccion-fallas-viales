UV ?= uv
UVX ?= uvx
PYTHON ?= $(UV) run python
SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help setup lint lint-web lock-check imports types test test-base test-full \
	fixtures data train eval reproduce serve build docs ci-build ci-installer ci-local

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Install the locked environment
	$(UV) sync

lint: ## Run the linter and the formatter check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

lint-web: ## Run Biome on the web modules (skipped when Node is absent)
	@if command -v npx >/dev/null 2>&1; then \
		npx --yes @biomejs/biome@2.6.8 ci src/rdv/entrypoints/web/static; \
	else \
		echo "npx not found: skipping Biome"; \
	fi

lock-check: ## Assert that uv.lock is up to date
	$(UV) lock --check

imports: ## Check the hexagonal import contracts
	$(UV) run lint-imports

types: ## Run the strict type checker
	$(UV) run mypy --strict src

test: ## Run the test suite
	$(UV) run pytest -q

test-base: ## Run the torch-free matrix test command
	$(UV) run pytest -q -m "not train and not ui and not parity and not realmodel"

test-full: ## Run the full suite with the coverage gate
	$(UV) run pytest -q -m "not realmodel" --cov=src/rdv --cov-report=term-missing --cov-fail-under=85

fixtures: ## Regenerate the committed test fixtures
	$(PYTHON) scripts/make_fixtures.py

data: ## Download the RDD2022 countries used for training
	$(UV) run rdv data download --countries Japan India Czech

train: ## Fine-tune the detector from the pinned configuration
	$(UV) run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml

eval: ## Evaluate the exported model on the held-out test split
	$(UV) run rdv evaluate --data "$${RDV_DATA_DIR}/rdd2022" --splits "$${RDV_DATA_DIR}/rdd2022/splits.json" --split test --out reports/v1

reproduce: ## Re-run the documented data preparation and evaluation (D23, docs/TRAINING.md)
	RDV_DATA_DIR=data $(UV) run rdv data download --countries Japan India Czech
	RDV_DATA_DIR=data $(UV) run rdv data download --ood-norway 1000
	RDV_DATA_DIR=data $(UV) run rdv data verify
	RDV_DATA_DIR=data $(UV) run rdv data prepare
	RDV_DATA_DIR=data $(UV) run rdv evaluate --data data/rdd2022 --splits data/rdd2022/splits.json --split test --model artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx --op-threshold auto --ood data/rdd2022-ood-norway --out reports/v1

serve: ## Start the local web app
	$(UV) run rdv serve

build: ## Build the wheel and the source distribution
	$(UV) build

docs: ## Check every relative link and anchor in README/docs
	$(UV) run pytest tests/unit/test_docs_links.py -q

ci-build: ## Build and inspect the distributions like the CI build job
	rm -rf dist
	$(UV) build
	$(UVX) twine check dist/*
	$(PYTHON) -c "import glob, zipfile; names = zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); assert any(n.endswith('assets/models.json') for n in names); assert any('/assets/samples/' in n and n.endswith('.jpg') for n in names); assert any(n.endswith('static/index.html') for n in names); assert not any(n.endswith('.onnx') for n in names); assert not any(n.startswith('tests/') for n in names); print('OK: wheel contents')"

ci-installer: ci-build ## Install the wheel in a clean tool dir and run the quickstart
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	UV_TOOL_DIR="$$tmp/tools" UV_TOOL_BIN_DIR="$$tmp/bin" $(UV) tool install --force ./dist/*.whl; \
	"$$tmp/bin/rdv" --version; \
	"$$tmp/bin/rdv" doctor --json > "$$tmp/doctor.json"; \
	"$$tmp/bin/rdv" detect tests/fixtures/rdd2022-mini/Japan/train/images/Japan_002986.jpg \
		--model tests/fixtures/models/tiny-detector.onnx --out "$$tmp/out"; \
	test -f "$$tmp/out/Japan_002986.detections.json"; \
	"$$tmp/bin/rdv" serve --no-browser --port 16399 \
		--model tests/fixtures/models/tiny-detector.onnx > "$$tmp/serve.log" 2>&1 & \
	pid=$$!; \
	ready=0; \
	for _ in $$(seq 1 40); do \
		if curl -fsS "http://127.0.0.1:16399/api/health" -o "$$tmp/health.json"; then ready=1; break; fi; \
		sleep 1; \
	done; \
	kill "$$pid" 2>/dev/null || true; \
	wait "$$pid" 2>/dev/null || true; \
	test "$$ready" = "1"; \
	grep -q '"status":"ok"' "$$tmp/health.json"; \
	echo "OK: clean-install quickstart"

ci-local: ## Run every CI job command locally (skips the OS matrix only)
	$(UV) sync --locked --extra train --extra cpu
	$(MAKE) lock-check
	$(MAKE) lint
	$(MAKE) lint-web
	$(MAKE) imports
	$(MAKE) types
	$(MAKE) test-base
	$(UV) run playwright install chromium
	$(MAKE) test-full
	$(MAKE) docs
	$(MAKE) ci-installer
