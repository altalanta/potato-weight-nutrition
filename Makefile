.PHONY: setup run teaser test lint clean help infra.init infra.plan infra.apply infra.destroy infra.output

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

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

# Infrastructure Management
infra.init:  ## Initialize Terraform
	cd infra/terraform && terraform init

infra.plan:  ## Plan Terraform changes (PROJECT_PREFIX=cdp-dev AWS_REGION=us-east-1)
	cd infra/terraform && terraform plan \
		-var="project_prefix=$(or $(PROJECT_PREFIX),cdp-dev)" \
		-var="aws_region=$(or $(AWS_REGION),us-east-1)" \
		-var="github_repo=$(shell git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')" \
		-var="environment=$(or $(ENVIRONMENT),dev)"

infra.apply:  ## Apply Terraform changes (PROJECT_PREFIX=cdp-dev AWS_REGION=us-east-1)
	cd infra/terraform && terraform apply \
		-var="project_prefix=$(or $(PROJECT_PREFIX),cdp-dev)" \
		-var="aws_region=$(or $(AWS_REGION),us-east-1)" \
		-var="github_repo=$(shell git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')" \
		-var="environment=$(or $(ENVIRONMENT),dev)"

infra.output:  ## Show Terraform outputs
	cd infra/terraform && terraform output

infra.destroy:  ## Destroy Terraform infrastructure (PROJECT_PREFIX=cdp-dev AWS_REGION=us-east-1)
	cd infra/terraform && terraform destroy \
		-var="project_prefix=$(or $(PROJECT_PREFIX),cdp-dev)" \
		-var="aws_region=$(or $(AWS_REGION),us-east-1)" \
		-var="github_repo=$(shell git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')" \
		-var="environment=$(or $(ENVIRONMENT),dev)"

infra.env:  ## Generate .env.aws file from Terraform outputs
	@echo "# Environment variables for potato-weight-nutrition pipeline" > .env.aws
	@echo "# Generated from Terraform outputs on $$(date)" >> .env.aws
	@echo "" >> .env.aws
	@cd infra/terraform && terraform output -json env_vars | jq -r 'to_entries[] | "\(.key)=\(.value)"' >> ../../.env.aws
	@echo "" >> .env.aws
	@echo "# Additional configuration" >> .env.aws
	@echo "STORAGE_BACKEND=s3" >> .env.aws
	@echo ".env.aws file generated successfully!"

# Observability Management
obs.up:  ## Start observability stack (Grafana, Prometheus, Loki, Jaeger, OpenTelemetry)
	docker compose -f docker/docker-compose.yml up -d
	@echo "🚀 Observability stack started!"
	@echo "📊 Grafana UI: http://localhost:3000 (admin/admin)"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "📝 Loki: http://localhost:3100"
	@echo "🔍 Jaeger: http://localhost:16686"

obs.down:  ## Stop observability stack
	docker compose -f docker/docker-compose.yml down
	@echo "🛑 Observability stack stopped"

obs.logs:  ## Show observability stack logs
	docker compose -f docker/docker-compose.yml logs -f

obs.status:  ## Check observability stack status
	docker compose -f docker/docker-compose.yml ps

obs.clean:  ## Remove observability stack and volumes (destructive)
	docker compose -f docker/docker-compose.yml down -v
	docker system prune -f
	@echo "🧹 Observability stack cleaned"

# Lineage Management  
lineage.up:  ## Start Marquez lineage tracking
	docker compose -f openlineage/marquez-docker-compose.yml up -d
	@echo "🔗 Marquez lineage stack started!"
	@echo "🌐 Marquez Web UI: http://localhost:3000"
	@echo "📡 Marquez API: http://localhost:5000"

lineage.down:  ## Stop Marquez lineage tracking
	docker compose -f openlineage/marquez-docker-compose.yml down
	@echo "🛑 Marquez lineage stack stopped"

lineage.clean:  ## Remove Marquez stack and data (destructive)
	docker compose -f openlineage/marquez-docker-compose.yml down -v
	@echo "🧹 Marquez lineage stack cleaned"

# Data Contracts
contracts.validate:  ## Run Great Expectations validation
	@echo "🔍 Running data contracts validation..."
	@if [ -f "great_expectations/great_expectations.yml" ]; then \
		great_expectations checkpoint run bronze_nutrition_checkpoint; \
		great_expectations checkpoint run silver_features_checkpoint; \
	else \
		echo "⚠️  Great Expectations not yet configured"; \
		echo "📋 Example expectations available in data_contracts/expectations/"; \
	fi

# Repro Management
repro:  ## Run reproducible pipeline end-to-end with observability
	@echo "🔬 Starting reproducible pipeline run..."
	@$(MAKE) obs.up
	@$(MAKE) lineage.up
	@sleep 10  # Wait for services to be ready
	@echo "🏃 Running pipeline with observability..."
	@export OPENLINEAGE_URL=http://localhost:5000 && \
	 export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 && \
	 PYTHONPATH=src python -m potato_pipeline.cli pipeline
	@echo "📊 Generating reproducible run report..."
	@mkdir -p reports/repro_run
	@$(MAKE) infra.env || echo "⚠️  Infrastructure not deployed, skipping AWS env"
	@echo "✅ Reproducible run completed!"
	@echo "📊 View results at: http://localhost:3000"
	@echo "🔗 View lineage at: http://localhost:3000 (Marquez)"

# Development helpers
check: lint test  ## Run both linting and tests

all: setup teaser run test  ## Run complete setup and pipeline