import argparse
import os
import re
from typing import Union, Any, Dict, List, Union

import dotenv
import prompt_templates
from config import LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS
from huggingface_hub import login
from lib import read_jsonl, write_jsonl
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams


dotenv.load_dotenv()
SYSTEM_PROMPT = prompt_templates.default_system_prompt

DATASET_TO_FILE_NAME = {
    "2wikimultihopqa": "nf_test_79_filtered.jsonl",
    "hotpotqa": "nf_test_62_filtered.jsonl",
    "musique": "nf_test_82_filtered.jsonl",
    "nq": "nf_test_126_filtered.jsonl",
    "squad": "nf_test_300_filtered.jsonl",
    "trivia": "nf_test_296_filtered.jsonl",
}

MODEL_PATH_TO_ALIAS = {
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-ins",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-ins",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b-ins",
    "meta-llama/Meta-Llama-3.2-3B-Instruct": "llama-3.2-3b-ins",
}


def make_prompt_batch(
    prompt_template, input_instances: List[Dict[str, Any]], use_gpt: bool = False
) -> Union[List[List[Dict[str, str]]], List[str]]:
    prompt_batch = []
    for instance in input_instances:
        question_text = instance["question_text"]
        answer = (
            instance["answers_objects"][0]["spans"][0]
            if len(instance["answers_objects"][0]["spans"]) == 1
            else str(instance["answers_objects"][0]["spans"])
        )

        input_prompt = prompt_template.format(question=question_text, ground0=answer)

        if use_gpt:  # use OpenAI API
            formatted_chat_template = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_prompt},
            ]
            prompt_batch.append(formatted_chat_template)
        else:
            prompt_batch.append(input_prompt)

    return prompt_batch


def extract_references_from_model_output(
    model_output: str,
) -> Dict[str, Union[str, None]]:
    model_output = model_output.strip()

    answer1_regex = r"([Aa]nswer\s*1\s*:\s*)(.*)(?=\n)"
    answer2_regex = r"([Aa]nswer\s*2\s*:\s*)(.*)(?=\n)"
    answer3_regex1 = r"([Aa]nswer\s*3\s*:\s*)(.*)(?=\n)"
    answer3_regex2 = r"([Aa]nswer\s*3\s*:\s*)(.*)(?=$)"

    if re.match(answer1_regex, model_output):
        answer1 = re.search(answer1_regex, model_output).group(2).strip()
        model_output = re.sub(answer1_regex, "", model_output)
    else:
        answer1 = None

    print(re.match(answer2_regex, model_output))
    if re.match(answer2_regex, model_output):
        answer2 = re.search(answer2_regex, model_output).group(2).strip()
        model_output = re.sub(answer2_regex, "", model_output)
    else:
        answer2 = None

    if re.match(answer3_regex1, model_output):
        answer3 = re.search(answer3_regex1, model_output).group(2).strip()
    elif re.match(answer3_regex2, model_output):
        answer3 = re.search(answer3_regex2, model_output).group(2).strip()
    else:
        answer3 = None

    references = {"answer1": answer1, "answer2": answer2, "answer3": answer3}

    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path.",
        choices=(
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "gpt-4o-mini-2024-07-18",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3.2-3B-Instruct",
            "gpt-4o-2024-08-06",
            "gpt-3.5-turbo-16k",
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset name.",
        choices=(
            "hotpotqa",
            "2wikimultihopqa",
            "musique",
            "nq",
            "trivia",
            "squad",
            "webglmqa",
        ),
    )
    parser.add_argument(
        "--file_name",
        type=str,
        help="file name. (e.g., `nf_test_300_filtered.jsonl`)",
        default=None,
    )
    args = parser.parse_args()

    if args.file_name is None:
        args.file_name = DATASET_TO_FILE_NAME[args.dataset_name]

    model_alias = MODEL_PATH_TO_ALIAS.get(args.model_path, args.model_path)

    output_folder = os.path.join(
        "data", "reference_list_construction", args.dataset_name
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filepath = os.path.join(output_folder, args.file_name)
    if os.path.exists(output_filepath):
        input_instances = read_jsonl(
            output_filepath
        )  # read the output file as input to continue the process
    else:
        input_filepath = os.path.join(
            "data", "processed_data", args.dataset_name, args.file_name
        )
        input_instances = read_jsonl(input_filepath)

    output_instances = []

    # For generating the highest standard reference answer
    if args.model_path == "gpt-4o-2024-08-06":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=openai_api_key)

        prompt_template = (
            prompt_templates.generating_highest_reference
        )  # prompt for generating the highest standard reference answer
        prompt_batch = make_prompt_batch(prompt_template, input_instances, use_gpt=True)

        for instance, formatted_chat_template in zip(
            tqdm(input_instances), prompt_batch
        ):
            if "model_output_for_reference_generation" not in instance:
                instance["model_output_for_reference_generation"] = {}
            model_output_dict = instance["model_output_for_reference_generation"]

            response = openai_client.chat.completions.create(
                model=args.model_path,
                messages=formatted_chat_template,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
            )
            output = response.choices[0].message.content.strip()
            model_output_dict[model_alias] = output
            instance["model_output_for_reference_generation"] = model_output_dict

            output_instances.append(instance)

    # For generating other reference answers in R sorted by quality descendingly
    else:
        # Use OpenAI API for GPT-4o-mini-2024-07-18 and GPT-3.5-turbo-16k
        if (
            args.model_path == "gpt-4o-mini-2024-07-18"
            or args.model_path == "gpt-3.5-turbo-16k"
        ):
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            openai_client = OpenAI(api_key=openai_api_key)

            prompt_template = (
                prompt_templates.generating_other_references
            )  # prompt for generating other reference answers
            prompt_batch = make_prompt_batch(
                prompt_template, input_instances, use_gpt=True
            )

            for instance, formatted_chat_template in zip(
                tqdm(input_instances), prompt_batch
            ):
                if "model_output_for_reference_generation" not in instance:
                    instance["model_output_for_reference_generation"] = {}
                model_output_dict = instance["model_output_for_reference_generation"]

                response = openai_client.chat.completions.create(
                    model=args.model_path,
                    messages=formatted_chat_template,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    max_tokens=LLM_MAX_TOKENS,
                )
                output = response.choices[0].message.content.strip()
                model_output_dict[model_alias] = output
                instance["model_output_for_reference_generation"] = model_output_dict

                output_instances.append(instance)

        # Use VLLM for Mistral-7B-Instruct-v0.2, Meta-Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct, and Meta-Llama-3.1-8B-Instruct
        elif (
            args.model_path == "mistralai/Mistral-7B-Instruct-v0.2"
            or args.model_path == "meta-llama/Meta-Llama-3-8B-Instruct"
            or args.model_path == "meta-llama/Meta-Llama-3.1-8B-Instruct"
            or args.model_path == "meta-llama/Meta-Llama-3.2-3B-Instruct"
        ):
            huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
            login(token=huggingface_token)

            prompt_template = (
                prompt_templates.generating_other_references
            )  # prompt for generating other reference answers
            prompt_batch = make_prompt_batch(
                prompt_template, input_instances, use_gpt=False
            )

            device = "cuda"
            sampling_params = SamplingParams(
                temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, max_tokens=LLM_MAX_TOKENS
            )
            llm = LLM(model=args.model_path, device=device)
            output_list = llm.generate(prompt_batch, sampling_params)

            for instance, output in zip(
                tqdm(input_instances, desc="Generating references"), output_list
            ):
                if "model_output_for_reference_generation" not in instance:
                    instance["model_output_for_reference_generation"] = {}
                model_output_dict = instance["model_output_for_reference_generation"]

                model_output_dict[model_alias] = output.outputs[0].text
                instance["model_output_for_reference_generation"] = model_output_dict

                output_instances.append(instance)

    write_jsonl(output_instances, output_filepath)


if __name__ == "__main__":
    main()
