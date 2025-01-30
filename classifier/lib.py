import json
from typing import List, Dict


def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf8", errors="ignore") as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [
            json.loads(line.strip()) for line in file.readlines() if line.strip()
        ]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def get_nfqa_category_prediction(nfqa_model, nfqa_tokenizer, text, device="cuda"):
    output = nfqa_model(**nfqa_tokenizer(text, return_tensors="pt").to(device))
    index = output.logits.argmax()
    return nfqa_model.config.id2label[int(index)]


CORPUS_NAME_DICT = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "nq": "wiki",
    "trivia": "wiki",
    "squad": "wiki",
    "ms_marco": None,
}
