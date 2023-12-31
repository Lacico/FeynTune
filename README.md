## Installation

A ready-made environment is provided with Docker. So, on Linux or WSL(v2), make sure you have installed [Docker](https://docs.docker.com/engine/install/) with [Docker Compose](https://docs.docker.com/compose/install/).

Commands for installation and environment management have been set up in the [Makefile](./Makefile). 

Install environment (GPU support included):

```sh
make install
```

### Launch container and start jupyter lab running in environment

To launch (and start a notebook server in the background) you can run (CPU only):

```sh
make up
```

Or:

```sh
make up-gpu
```

Once it's launched, retrieve the notebook link by running `make logs` (can take a few seconds to appear in the container log).

### Development

A [.devcontainer](./.devcontainer) is provided, which should allow you to properly develop with full IDE support form inside the container in VSCode. You can enable this by installing VSCode Remote Containers extension and choosing "open project in container".


## Libraries

[HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index)

[DeepSpeed](https://github.com/microsoft/DeepSpeed)

[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

[PEFT](https://github.com/huggingface/peft)

[GitHub: QLoRA](https://github.com/artidoro/qlora)

[FastChat](https://github.com/lm-sys/FastChat)

[SpQR paper implementation](https://github.com/Vahe1994/SpQR)

[Llama Recipes](https://github.com/facebookresearch/llama-recipes)

[GGML (post-training quantization and inference, CPU-focused)](https://github.com/ggerganov/ggml)

[GPTQ (post-training quantization, GPU)](https://github.com/ist-daslab/gptq)

[Fine Tuning Language Models with Just Forward Passes (code)](https://github.com/princeton-nlp/mezo)

## Videos

[Understanding 4bit Quantization: QLoRA explained (w/ Colab)](https://www.youtube.com/watch?v=TPcXVJ1VSRI)

[PEFT LoRA Explained in Detail - Fine-Tune your LLM on your local GPU](https://www.youtube.com/watch?v=YVU5wAA6Txo)

[Boost Fine-Tuning Performance of LLM: Optimal Architecture w/ PEFT LoRA Adapter-Tuning on Your GPU](https://www.youtube.com/watch?v=A-a-l_sFtYM)

## Blog Posts

[Anyscale Blog: Fine-tuning Llama 2](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications)

[Medium: Easily Finetune Llama 2 for Your Text-to-SQL Applications](https://medium.com/llamaindex-blog/easily-finetune-llama-2-for-your-text-to-sql-applications-ecd53640e10d)

[GitHub: Modal Finetune SQL Tutorial](https://github.com/run-llama/modal_finetune_sql/blob/main/tutorial.ipynb)

[Medium: Easily Finetune Llama 2 for Your Text-to-SQL Applications](https://medium.com/llamaindex-blog/easily-finetune-llama-2-for-your-text-to-sql-applications-ecd53640e10d)

[GitHub: Ray Project - Finetuning LLMS with Deepspeed](https://github.com/ray-project/ray/tree/workspace_templates_2.6.1/doc/source/templates/04_finetuning_llms_with_deepspeed)


[Codehammer: How to Load Llama 13B for Inference on a Single RTX 4090](https://codehammer.io/how-to-load-llama-13b-for-inference-on-a-single-rtx-4090/)

[Storm in the Castle: Alpaca 13B](https://www.storminthecastle.com/posts/alpaca_13B/)


## Papers

[QLoRA paper](https://arxiv.org/abs/2305.14314)

[SpQR - Sparse-Quantized Representation (to be integrated in BitsAndBytes)](https://arxiv.org/pdf/2306.03078.pdf)

[Fine Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)

## Running Code

You will need to export an environment variable: `HUGGINGFACE_KEY=<your_key>` and then you can use the build, run and lint commands defined in the makefile e.g. `make build`. The docker container uses multiple stages to cache models from huggingface such that they aren't downloaded for every code change. To add more models to the cache simply add a line to the `ModelPaths` class in the [model_paths.py](./src/model_paths.py) module.

