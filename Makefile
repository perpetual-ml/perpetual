SHELL := /bin/bash
PYTHON := $(shell command -v python3 2>/dev/null || command -v python)
.PHONY: help

help: ## List all options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

init: ## Initialise local resources for testing the software
	@cp "README.md" "python-package/README.md"
	@cp "LICENSE" "python-package/LICENSE"
	
	@cd python-package && $(PYTHON) -m pip install -e .[dev]

	@$(PYTHON) -m pip install pandas seaborn
	@$(PYTHON) scripts/make_resources.py

	@rm "python-package/README.md"
	@rm "python-package/LICENSE"

	@$(PYTHON) -m pip install pre-commit
	@pre-commit install
	
clean: ## Clean the project using cargo
	cargo clean

purge: ## Delete everything that is NOT tracked by git
	git clean -d -x -f

build: ## Build the project using cargo
	cargo build

lint: ## Lint the project using cargo
	@rustup component add clippy 2> /dev/null
	cargo clippy

py-test: ## Test the projects Python implementation
	source scripts/run-python-tests.sh

rust-test: ## Test the projects Rust implementation
	cargo test

test: ## Test the project and all its implementations
	cargo test
	source scripts/run-python-tests.sh

fmt: ## Format the project using cargo
	@rustup component add rustfmt 2> /dev/null
	cargo fmt