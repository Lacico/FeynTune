from peft import LoraConfig, get_peft_model
import torch
from model_paths import ModelPaths
from models import get_model
from transformers import AutoTokenizer
from os import getenv
from datasets import load_dataset
import bitsandbytes as bnb
from trl import SFTTrainer

max_length = 512
batch_size = 128
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size


def get_tokenizer(path: str, token: str):
    tokenizer = AutoTokenizer.from_pretrained(ModelPaths.LLAMA_2_7B, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_dataset():
    # Load the databricks dataset from Hugging Face

    return load_dataset("wikisql", split="train")


def create_peft_config(modules):
    return LoraConfig(
        r=1,
        lora_alpha=16,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_peft_config(model):
    modules = find_all_linear_names(model)
    return create_peft_config(modules)


def train(model, tokenizer, dataset, output_dir):
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="question",
    )
    trainer.train()
    # Saving model
    print("Saving last checkpoint of the model...")
    import os

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(
        f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} "
    )
    return trainable_model_params


def main(token: str):
    model = get_model(ModelPaths.LLAMA_2_7B, token=token)
    tokenizer = get_tokenizer(ModelPaths.LLAMA_2_7B, token=token)
    dataset = get_dataset()

    ori_p = print_number_of_trainable_model_parameters(model)
    model = get_peft_model(model, get_peft_config(model))

    peft_p = print_number_of_trainable_model_parameters(model)
    print(
        f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}"
    )
    output_dir = "/results"
    train(model, tokenizer, dataset, output_dir)


if __name__ == "__main__":
    token = getenv("HUGGINGFACE_KEY")
    if not token:
        raise RuntimeError("Set the HUGGINGFACE_KEY environment variable")
    main(token)
