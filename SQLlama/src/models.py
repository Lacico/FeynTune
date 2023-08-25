from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from model_paths import ModelPaths

# nf4" use a symmetric quantization scheme with 4 bits precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def get_model(path: str, token: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",
    )
