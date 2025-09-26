SHELL := /bin/bash
PYTHON := $(shell command -v python3 2>/dev/null || command -v python)
VENV_DIR := .venv
# Determine the correct activation script path based on the OS
ifeq ($(OS), Windows_NT)
    VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
else
    VENV_ACTIVATE := $(VENV_DIR)/bin/activate
endif

.PHONY: help venv clean purge build lint py-test rust-test test fmt

# This target ensures the virtual environment exists before any python commands run.
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment with uv..."; \
		$(PYTHON) -m uv venv $(VENV_DIR); \
	fi


help: ## List all options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'


init: venv ## Initialise local resources for testing the software
	@echo "Installing Python dependencies..."
	@cp "README.md" "python-package/README.md"
	@cp "LICENSE" "python-package/LICENSE"
	@source $(VENV_ACTIVATE) && \
	    cd python-package && \
	    uv pip install -e ".[dev]" && \
	    cd .. && \
	    uv pip install pandas seaborn && \
	    python scripts/make_resources.py

	@echo "Installing pre-commit hooks..."
	@source $(VENV_ACTIVATE) && \
	    uv pip install pre-commit && \
	    pre-commit install


clean: ## Clean the project using cargo
	cargo clean


purge: ## Delete everything that is NOT tracked by git
	git clean -d -x -f


build: ## Build the project using cargo
	cargo build


lint: ## Lint the project using cargo
	@rustup component add clippy 2> /dev/null
	cargo clippy


py-test-sh: venv ## Test the projects Python implementation
	@echo "Running Python tests..."
	@source $(VENV_ACTIVATE) && \
	    bash scripts/run-python-tests.sh


py-test-ps: venv ## Test the projects Python implementation
	@echo "Running Python tests..."
	@source $(VENV_ACTIVATE) && \
	    powershell.exe -ExecutionPolicy Bypass -File scripts/run-python-tests.ps1


py-test-ps-single: venv ## Test the projects Python implementation
	@echo "Running Python tests..."
	@source $(VENV_ACTIVATE) && \
	    powershell.exe -ExecutionPolicy Bypass -File scripts/run-single-python-test.ps1


rust-test: ## Test the projects Rust implementation
	cargo test


test: venv ## Test the project and all its implementations
	@echo "Running both Rust and Python tests..."
	cargo test
	@source $(VENV_ACTIVATE) && \
	    bash scripts/run-python-tests.sh


fmt: ## Format the project using cargo
	@rustup component add rustfmt 2> /dev/null
	cargo fmt
