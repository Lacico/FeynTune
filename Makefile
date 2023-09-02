inference-host ?= 0.0.0.0
inference-port ?= 6969
inference-model ?= bigscience/bloomz-560m

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

.PHONY: help format lint test test-gpu test-all up up-gpu down stop logs
help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

build: split-requirements-trn add-vllm-submodule build-base build-inf build-trn

clean: clean-trn clean-docker

clean-docker:
	docker compose down
	chmod 755 bin/clean_docker_images.sh && sh bin/clean_docker_images.sh

clean-trn:
	rm -f ./train/pyproject-core.toml

# use system python to split poetry requirements into "core" and remaining
# by parsing toml to workaround docker build caching
split-requirements-trn:
	@pip install toml
	@if [ ! -f ./train/pyproject-core.toml ]; then \
		python bin/split_dependencies.py; \
	fi

add-vllm-submodule:
	if [ ! -d "./inference/vllm/.git" ]; then \
		git rm --cached inference/vllm; \
		git submodule add https://github.com/vllm-project/vllm inference/vllm; \
	fi

build-base: 
	docker build -t feyntune-base -f Dockerfile.base .

build-inf: build-base ## Build the Docker images (gpu/cpu currently same dockerfile)
	docker compose build inference-cpu
	docker compose build inference-gpu

build-trn: build-base
	docker compose build train-cpu --no-cache
	docker compose build train-gpu

lint: lint-inf lint-trn

lint-inf:
	docker compose run --rm inference-cpu poetry run mypy .
	docker compose run --rm inference-cpu poetry run black . --check
	docker compose run --rm inference-cpu poetry run ruff check .

lint-trn:
	docker compose run --entrypoint "" --rm train-gpu poetry run black . --check

format: format-inf format-trn

format-inf:
	docker compose run --rm inference-cpu poetry run black .

format-trn:
	docker compose run --entrypoint "" --rm train-gpu poetry run black .

start-inf-api-gpu:
	docker-compose run --rm inference-gpu python -m vllm.entrypoints.openai.api_server --host $(inference-host) --port $(inference-port) --model $(inference-model)

start-inf-api-cpu:
	docker-compose run --rm inference-cpu python -m vllm.entrypoints.openai.api_server --host $(inference-host) --port $(inference-port) --model $(inference-model)

# test: ## Run tests (CPU only)
# 	docker compose run --rm cpu poetry run pytest -v -m "not gpu"

# test-gpu: ## Run tests (GPU)
# 	docker compose run --rm gpu poetry run pytest -v -m gpu

# test-all: test test-gpu ## Run all tests
