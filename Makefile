# neural-decoding-infra Makefile
#
# Common development commands. Run `make help` for a list.

.PHONY: help install install-dev install-cluster install-all-extras \
        install-brainomni install-large-lm lint format test test-unit test-cov \
        smoke smoke-tier0 smoke-tier1 clean docs-build pre-commit

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ────────────────────────── Install ──────────────────────────

install: ## Install the package (core deps only)
	$(PIP) install -e .

install-dev: ## Install with dev/test deps (no heavy extras)
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all-extras: ## Install EVERYTHING (cluster-side; all extras)
	$(PIP) install -e ".[all,dev]"

install-brainomni: ## Install with BrainOmni extra only
	$(PIP) install -e ".[brainomni,dev]"

install-large-lm: ## Install with large-LM extra (vLLM + FlashAttention)
	$(PIP) install -e ".[large-lm,dev]"

install-cluster: ## Full cluster install (conda env create + pip install [all])
	conda env create -f environment.yml || conda env update -f environment.yml
	$(PIP) install -e ".[all,dev]"
	pre-commit install

# ────────────────────────── Lint / Format ──────────────────────────

lint: ## Run ruff + mypy on the codebase
	ruff check .
	ruff format --check .
	mypy recon/

format: ## Auto-format code
	ruff check --fix .
	ruff format .

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

# ────────────────────────── Test ──────────────────────────

test: test-unit ## Alias for test-unit

test-unit: ## Run unit tests (CPU only)
	pytest tests/unit/ -v

test-cov: ## Run unit tests with coverage report
	pytest tests/unit/ --cov=recon --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests (may need data)
	pytest tests/integration/ -v

# ────────────────────────── Cluster smoke ──────────────────────────

smoke: smoke-tier0 ## Alias for smoke-tier0

smoke-tier0: ## Run Tier 0 smoke (fake data, 1 node)
	bash scripts/smoke.sh --tier=L0

smoke-tier1: ## Run Tier 1 smoke (real data, 1 node)
	bash scripts/smoke.sh --tier=L1

smoke-all: ## Run all smoke tests
	bash scripts/smoke.sh --tier=L0
	bash scripts/smoke.sh --tier=L1

# ────────────────────────── Cluster operations ──────────────────────────

launch-multi: ## Launch multi-node training (e.g., make launch-multi CONFIG=baseline)
	bash scripts/launch_multi_node.sh $(CONFIG)

decode: ## Decode one story (override CKPT, SUB, STORY)
	python -m recon.cli.decode \
		--checkpoint $(CKPT) \
		--subject $(SUB) \
		--story $(STORY) \
		--output $(OUT)

# ────────────────────────── Docs ──────────────────────────

docs-build: ## Build documentation site (when configured)
	@echo "Documentation site build: not yet configured (docs are markdown-only for now)"

# ────────────────────────── Cleanup ──────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete