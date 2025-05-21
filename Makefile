SHELL := /bin/bash
.PHONY: help

help: ## List all options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: ## Clean the project using cargo
	cargo clean

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