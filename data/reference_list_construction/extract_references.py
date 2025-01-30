import argparse
import os
import re
from typing import Union, Any, Dict, List, Union

from lib import read_jsonl, write_jsonl
from tqdm import tqdm

INPUT_DATASET_TO_FILE_NAME = {
    "2wikimultihopqa": "nf_test_79_filtered.jsonl",
    "hotpotqa": "nf_test_62_filtered.jsonl",
    "musique": "nf_test_82_filtered.jsonl",
    "nq": "nf_test_126_filtered.jsonl",
    "squad": "nf_test_300_filtered.jsonl",
    "trivia": "nf_test_296_filtered.jsonl",
}

OUTPUT_DATASET_TO_FILE_NAME = {
    "2wikimultihopqa": "odqa_nf_test.jsonl",
    "hotpotqa": "odqa_nf_test.jsonl",
    "musique": "odqa_nf_test.jsonl",
    "nq": "odqa_nf_test.jsonl",
    "squad": "odqa_nf_test.jsonl",
    "trivia": "odqa_nf_test.jsonl",
}

MODEL_LIST = [
    "gpt-4o-2024-08-06",
    "gpt-3.5-turbo-16k",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

MODEL_PATH_TO_ALIAS = {
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-ins",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-ins",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b-ins",
}


def extract_references_from_model_output(
    model_output: str,
) -> Dict[str, Union[str, None]]:
    model_output = model_output + "\n"

    answer1_regex1 = (
        r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*1\s*:\s*)(.*)(?=\n*[Aa][Nn][Ss][Ww][Ee][Rr]\s*2)"
    )
    answer1_regex2 = (
        r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*1\s*:\s*)(.*)(?=\s*[Aa][Nn][Ss][Ww][Ee][Rr]\s*2)"
    )
    answer1_regex3 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*1\s*:\s*)(.*)(?=\n)"

    answer2_regex1 = (
        r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*2\s*:\s*)(.*)(?=\n*[Aa][Nn][Ss][Ww][Ee][Rr]\s*3)"
    )
    answer2_regex2 = (
        r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*2\s*:\s*)(.*)(?=\s*[Aa][Nn][Ss][Ww][Ee][Rr]\s*3)"
    )
    answer2_regex3 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*2\s*:\s*)(.*)(?=\n)"

    answer3_regex1 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*3\s*:\s*)(.*)(?=Output:\s*\`{3,})"
    answer3_regex2 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*3\s*:\s*)(.*)(?=\`{3,})"
    answer3_regex3 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*3\s*:\s*)(.*)(?=\n)"
    answer3_regex4 = r"([Aa][Nn][Ss][Ww][Ee][Rr]\s*3\s*:\s*)(.*)(?=$)"

    answer1 = None
    for regex in [answer1_regex1, answer1_regex2, answer1_regex3]:
        if re.search(regex, model_output):
            answer1 = re.search(regex, model_output).group(2).strip()
            model_output = re.sub(regex, "", model_output)
            break

    answer2 = None
    for regex in [answer2_regex1, answer2_regex2, answer2_regex3]:
        if re.search(regex, model_output):
            answer2 = re.search(regex, model_output).group(2).strip()
            model_output = re.sub(regex, "", model_output)
            break

    answer3 = None
    for regex in [answer3_regex1, answer3_regex2, answer3_regex3, answer3_regex4]:
        if re.search(regex, model_output):
            answer3 = re.search(regex, model_output).group(2).strip()
            break

    references = {"answer1": answer1, "answer2": answer2, "answer3": answer3}

    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset name.",
        choices=("hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"),
    )
    parser.add_argument(
        "--input_file_name",
        type=str,
        help="file name. (e.g., `nf_test_300_filtered.jsonl`)",
        default=None,
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="file name. (e.g., `odqa_nf_test.jsonl`)",
        default=None,
    )
    args = parser.parse_args()

    model_alias_list = [
        MODEL_PATH_TO_ALIAS[args.model_path] for args.model_path in MODEL_LIST
    ]

    if args.input_file_name is None:
        args.input_file_name = INPUT_DATASET_TO_FILE_NAME[args.dataset_name]
    if args.output_file_name is None:
        args.output_file_name = OUTPUT_DATASET_TO_FILE_NAME[args.dataset_name]

    input_folder = os.path.join(
        "data", "reference_list_construction", args.dataset_name
    )
    assert os.path.exists(input_folder), f"Input folder {input_folder} does not exist."
    input_filepath = os.path.join(input_folder, args.input_file_name)
    input_instances = read_jsonl(input_filepath)

    output_folder = os.path.join(
        "data", "reference_list_construction", args.dataset_name
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filepath = os.path.join(output_folder, args.output_file_name)
    output_instances = []

    # Extract references from model output
    for input_instance in tqdm(
        input_instances,
        desc=f"Extracting references from model output for {args.dataset_name}",
    ):
        output_instance = {}
        output_instance["question_id"] = input_instance["question_id"]
        output_instance["question_text"] = input_instance["question_text"]
        model_output_dict = input_instance["model_output_for_reference_generation"]
        ground_truth = input_instance["ground_truth"]

        answer_dict = {}
        answer_dict["ground_truth"] = {"answer1": ground_truth}

        for model_alias in model_alias_list:
            if model_alias == "gpt-4o":
                model_output = model_output_dict["gpt-4o"]
                answer_dict[model_alias] = {"answer1": model_output}
                continue

            else:
                model_output = model_output_dict[model_alias]
                references = extract_references_from_model_output(model_output)
                answer_dict[model_alias] = references

                if (
                    references["answer1"] is None
                    or references["answer1"].strip() == ""
                    or "```python" in references["answer1"]
                ):
                    del references["answer1"]
                if (
                    references["answer2"] is None
                    or references["answer2"].strip() == ""
                    or "```python" in references["answer2"]
                ):
                    del references["answer2"]
                if (
                    references["answer3"] is None
                    or references["answer3"].strip() == ""
                    or "```python" in references["answer3"]
                ):
                    del references["answer3"]

                answer_dict[model_alias] = references

        answer_list = [ground_truth]
        for model_alias in model_alias_list:
            if answer_dict[model_alias] is not None:
                for key, value in answer_dict[model_alias].items():
                    if value is not None:
                        answer_list.append(value)

        output_instance["num_answers"] = len(answer_list)
        output_instance["answer_dict"] = answer_dict
        output_instance["answer_list"] = answer_list
        output_instance["answer_label"] = []
        output_instances.append(output_instance)

    write_jsonl(output_instances, output_filepath)


if __name__ == "__main__":
    main()
