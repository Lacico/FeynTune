from transformers import AutoModelForCausalLM
from model_paths import ModelPaths
from os import getenv


def cache_models(token: str):
    [
        AutoModelForCausalLM.from_pretrained(ModelPaths.__dict__[p], token=token)
        for p in ModelPaths.__dict__
        if not callable(getattr(ModelPaths, p)) and not p.startswith("__")
    ]


if __name__ == "__main__":
    token = getenv("HUGGINGFACE_KEY")
    cache_models(token)
