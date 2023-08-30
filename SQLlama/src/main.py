from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import torch
from model_paths import ModelPaths
from models import get_model
from transformers import (
    AutoTokenizer,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from os import getenv
from data_loader import get_data_path
from datasets import load_dataset
from pathlib import Path
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cutoff_len = 512
batch_size = 1
max_steps: int = 200
learning_rate: float = 3e-4
eval_steps: int = 100
save_steps: int = 200
group_by_length: bool = True
gradient_accumulation_steps = 4
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def generate_prompt_sql(input, context, output=""):
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input}

### Context:
{context}

### Response:
{output}"""


def tokenize_(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        max_length=cutoff_len,
    )
    return result


def tokenize(prompt, tokenizer, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt_sql(
        data_point["input"],
        data_point["context"],
        data_point["output"],
    )
    input_prompt = generate_prompt_sql(data_point["input"], data_point["context"], "")
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    # tokenized_input_prompt = tokenize(input_prompt, tokenizer)
    return tokenized_full_prompt


def create_peft_config():
    return LoraConfig(
        r=1,
        lora_alpha=8,  # parameter for scaling
        target_modules=lora_target_modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(model, tokenizer, dataset, output_dir, val_set_size=200):
    gen_and_tokenize_prompt_partial = partial(
        generate_and_tokenize_prompt, tokenizer=tokenizer
    )

    if val_set_size > 0:
        train_val = dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(gen_and_tokenize_prompt_partial)
        val_data = train_val["test"].shuffle().map(gen_and_tokenize_prompt_partial)
    else:
        train_data = dataset["train"].shuffle().map(gen_and_tokenize_prompt_partial)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=False,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else "none",
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


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


def get_output_path():
    return Path("/results")


def main(token: str):
    model = get_model(ModelPaths.LLAMA_2_7B, token=token)
    tokenizer = AutoTokenizer.from_pretrained(
        ModelPaths.LLAMA_2_7B, add_eos_token=True, token=token
    )

    # Add new padding token to the tokenizes method
    # special_tokens_dict = {'pad_token': '<PAD>'}
    # tokenizer.add_special_tokens(special_tokens_dict)

    # Adjust the padding token ID in the tokenizer to reflect the new token
    # tokenizer.pad_token = '<PAD>'
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<PAD>')

    # Resize the model's token embeddings
    # model.resize_token_embeddings(len(tokenizer))

    # Initialize the padding token's embedding to zeros
    # padding_embedding = model.get_input_embeddings()
    # new_embedding = torch.zeros(1, padding_embedding.embedding_dim).to(device)
    # padding_embedding.weight.data = torch.cat([padding_embedding.weight.data, new_embedding], dim=0).to(device)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    # tokenizer.padding_side = 'right'
    # tokenizer.truncation_side = 'right'
    print(tokenizer)

    data_path = get_data_path().as_posix()
    data = load_dataset("json", data_files=data_path)

    ori_p = print_number_of_trainable_model_parameters(model)
    model = get_peft_model(model, create_peft_config())

    peft_p = print_number_of_trainable_model_parameters(model)
    print(
        f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}"
    )
    output_dir = get_output_path()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    train(model, tokenizer, data, output_dir)


if __name__ == "__main__":
    token = getenv("HUGGINGFACE_KEY")
    if not token:
        raise RuntimeError("Set the HUGGINGFACE_KEY environment variable")
    main(token)
