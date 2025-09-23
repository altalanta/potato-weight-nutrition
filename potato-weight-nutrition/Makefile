.PHONY: setup run teaser test lint clean help

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install dependencies and set up pre-commit hooks
	pip install -r requirements.txt
	pre-commit install
	@echo "Setup complete!"

run:  ## Run the complete analysis pipeline
	python -m potato_pipeline.cli pipeline

teaser:  ## Generate/refresh the teaser plot only
	python -m potato_pipeline.cli teaser

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run code linting with ruff
	ruff check src/ tests/
	ruff format --check src/ tests/

format:  ## Format code with ruff
	ruff format src/ tests/

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/

config:  ## Show current configuration
	python -m potato_pipeline.cli print-config

install:  ## Install package in development mode
	pip install -e .

# Development helpers
check: lint test  ## Run both linting and tests

all: setup teaser run test  ## Run complete setup and pipeline