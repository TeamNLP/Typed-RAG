import argparse
import os

import torch
from lib import read_jsonl, write_jsonl, get_nfqa_category_prediction
from model import RobertaNFQAClassification
from transformers import AutoTokenizer
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    choices=("hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"),
    help="dataset name.",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="train",
    choices=("train", "dev", "test_subsampled", "dev_500_subsampled"),
    help="",
)

args = parser.parse_args()


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nfqa_model = RobertaNFQAClassification.from_pretrained("Lurunchik/nf-cats").to(
        device
    )
    nfqa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    input_filepath = os.path.join(
        "data", "processed_data", args.dataset_name, f"{args.dataset_type}.jsonl"
    )
    output_filepath = os.path.join(
        "data", "processed_data", args.dataset_name, f"nf_{args.dataset_type}.jsonl"
    )

    input_instance = read_jsonl(input_filepath)

    output_instance = []
    for datum in tqdm(input_instance, desc=f"Predicting on {input_filepath}"):
        question_text = datum["question_text"]
        category_prediction = get_nfqa_category_prediction(
            nfqa_model, nfqa_tokenizer, question_text.strip(), device
        )

        # Save the prediction only if the category are non-factoid questions.
        if category_prediction == "NOT-A-QUESTION" or category_prediction == "FACTOID":
            continue

        datum["category_prediction"] = category_prediction
        output_instance.append(datum)

    write_jsonl(output_instance, output_filepath)


if __name__ == "__main__":
    main(args)
