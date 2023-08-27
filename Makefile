inference-host ?= 0.0.0.0
inference-port ?= 6969
inference-model ?= bigscience/bloomz-560m

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

.PHONY: help format lint test test-gpu test-all up up-gpu down stop logs
help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

build: build-base build-inference build-train

build-base: 
	docker build -t feyntune-base -f Dockerfile.base .

build-inference: build-base ## Build the Docker images (gpu/cpu currently same dockerfile)
	docker compose build inference-cpu
	docker compose build inference-gpu

build-train: build-base
	docker compose build train-cpu
	docker compose build train-gpu

# start-inference-gpu-nb:
# 	docker-compose run --rm inference-gpu poetry run poe jl

# start-inference-cpu-nb:
# 	docker-compose run --rm inference-cpu poetry run poe jl

# start-inference-gpu-api:
# 	docker-compose run --rm inference-gpu python -m vllm.entrypoints.openai.api_server --host $(inference-host) --port $(inference-port) --model $(inference-model)

# start-inference-cpu-api:
# 	docker-compose run --rm inference-cpu python -m vllm.entrypoints.openai.api_server --host $(inference-host) --port $(inference-port) --model $(inference-model)

# # install-gpu: install
# # 	docker compose build gpu

# format: ## Run code formatter: black
# 	docker compose run --rm cpu poetry run black .

# lint-inference: ## Run linters: mypy, black, ruff
# 	docker compose run --rm cpu poetry run mypy .
# 	docker compose run --rm cpu poetry run black . --check
# 	docker compose run --rm cpu poetry run ruff check .

# test: ## Run tests (CPU only)
# 	docker compose run --rm cpu poetry run pytest -v -m "not gpu"

# test-gpu: ## Run tests (GPU)
# 	docker compose run --rm gpu poetry run pytest -v -m gpu

# test-all: test test-gpu ## Run all tests

# up-gpu: ## Start the application
# 	docker compose up -d gpu

# up: 
# 	docker compose up -d cpu

# down:
# 	docker compose down 

# stop:
# 	docker compose stop cpu

# stop-gpu:
# 	docker compose stop gpu

# logs:  # get notebook url
# 	docker compose logs