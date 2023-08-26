from pathlib import Path
from datasets import load_dataset
import json


def get_data_path() -> Path:
    return Path("/outputs/data")


def load_data_sql():
    dataset = load_dataset("b-mc2/sql-create-context")

    dataset_splits = {"train": dataset["train"]}
    out_path = get_data_path()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for key, ds in dataset_splits.items():
        with open(out_path, "w") as f:
            for item in ds:
                newitem = {
                    "input": item["question"],
                    "context": item["context"],
                    "output": item["answer"],
                }
                f.write(json.dumps(newitem) + "\n")


if __name__ == "__main__":
    load_data_sql()
