import datasets
from datasets import load_dataset


def gen():
    data = load_dataset("lmms-lab/MMMU")
    yield from data["dev"]
    yield from data["validation"]


final_data = datasets.Dataset.from_generator(gen)

final_data.push_to_hub("pufanyi/MMMU", split="validation")
