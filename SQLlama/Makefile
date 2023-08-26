.PHONY: build build_cache finetune run lint

build_cache:
	@docker build . --build-arg HUGGINGFACE_KEY=${HUGGINGFACE_KEY} --target cache -t sqllama_cache

build:
	@docker build . --build-arg HUGGINGFACE_KEY=${HUGGINGFACE_KEY} --target finetuner -t sqllama_ft

finetune:
	@docker run -e HUGGINGFACE_KEY=${HUGGINGFACE_KEY} -v ${PWD}/checkpoints:/results --gpus all sqllama_ft:latest

run:
	@docker run -e HUGGINGFACE_KEY=${HUGGINGFACE_KEY} -v ${PWD}/checkpoints:/results --gpus all sqllama:latest  

lint:
	@black .
