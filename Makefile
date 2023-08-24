GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

.PHONY: help format lint test test-gpu test-all up up-gpu down stop logs
help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Build the Docker images (gpu/cpu currently same dockerfile)
	docker compose build cpu
	
# install-gpu: install
# 	docker compose build gpu

format: ## Run code formatter: black
	docker compose run --rm cpu poetry run black .

lint: ## Run linters: mypy, black, ruff
	docker compose run --rm cpu poetry run mypy .
	docker compose run --rm cpu poetry run black . --check
	docker compose run --rm cpu poetry run ruff check .

test: ## Run tests (CPU only)
	docker compose run --rm cpu poetry run pytest -v -m "not gpu"

test-gpu: ## Run tests (GPU)
	docker compose run --rm gpu poetry run pytest -v -m gpu

test-all: test test-gpu ## Run all tests

up-gpu: ## Start the application
	docker compose up -d gpu

up: 
	docker compose up -d cpu

down:
	docker compose down 

stop:
	docker compose stop cpu

stop-gpu:
	docker compose stop gpu

logs:  # get notebook url
	docker compose logs