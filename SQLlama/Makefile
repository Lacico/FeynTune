.PHONY: build run lint

build:
	@docker build . --build-arg HUGGINGFACE_KEY=${HUGGINGFACE_KEY} --target runner -t feyntune

run:
	@docker run -e HUGGINGFACE_KEY=${HUGGINGFACE_KEY} -v ${PWD}/checkpoints:/results --gpus all feyntune:latest  

lint:
	@black .
