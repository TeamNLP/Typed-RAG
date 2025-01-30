import argparse
import dotenv
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Tuple, Union, Optional

from config import (
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    VLLM_GPU_MEMORY_UTILIZATION,
)
from lib import read_jsonl, write_jsonl, read_json, write_json
from lib import make_single_request_dict
from lib import PRICING_PER_INPUT_TOKEN, PRICING_PER_OUTPUT_TOKEN
from openai import OpenAI
from prompt_templates import DEFAULT_SYS_PROMPT, LINKAGE_PROMPT_TEMPLATE
from tqdm import tqdm
from vllm import LLM, SamplingParams


dotenv.load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, help="random seed", default=42)
parser.add_argument(
    "--scorer_model_name",
    type=str,
    help="scorer_model_name",
    choices=("mistralai/Mistral-7B-Instruct-v0.2", "gpt-4o-mini-2024-07-18"),
)
parser.add_argument(
    "--model_alias_to_evaluate",
    type=str,
    help="model alias you want to evaluate",
    choices=(
        "gpt-4o-mini",
        "mistral-7b-ins",
        "llama-3.2-3b-ins",
        "llama-3.1-70b-ins",
    ),
)
parser.add_argument(
    "--num_references", type=int, help="number of reference answers", default=4
)
parser.add_argument(
    "--use_openai_batch_api",
    action="store_true",
    help="whether to use OpenAI Batch API",
)
parser.add_argument(
    "--openai_batch_api_mode",
    type=str,
    help="OpenAI Batch API mode",
    choices=("create", "retrieve", "cancel", "list"),
)
parser.add_argument(
    "--custom_id_prefix",
    type=str,
    help="custom id prefix for OpenAI Batch API",
    default="LINKAGE",
)
parser.add_argument(
    "--openai_batch_api_batch_id_to_cancel",
    type=str,
    help="OpenAI Batch API batch id to cancel",
)

args = parser.parse_args()


RANK_PATTERN = re.compile(r"\[([0-9]+)\]")
INT_TO_EN = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _make_linkage_prompt(
    question: str,
    reference_answer_list: List[str],
    candidate_answer: str,
    use_gpt: bool,
) -> Union[List[Dict[str, str]], str]:
    """
    Creates a formatted prompt for ranking the candidate answer against reference answers.

    Parameters:
    - question (str): The user's question.
    - reference_answer_list (list of str): List of reference answers in descending order of quality. (e.g., [best_answer, good_answer, average_answer, poor_answer])
    - candidate_answer (str): The answer to be ranked, generated using our RAG methodology, and intended for evaluation through the LINKAGE method.
    - use_gpt: bool - whether to use GPT

    Returns:
    - prompt: The formatted prompt {string or list of dictionaries}.
    """

    # Modified create_ground_reference() function in https://github.com/babyyang525/LINKAGE-Listwise-NFQA-Evaluation/blob/main/LINKAGE_method/LINKAGE.py
    # Format the reference answer list
    reference_answer_list_str = ""
    for idx, ref_ans in enumerate(reference_answer_list):
        reference_answer_list_str += f"Answer {idx + 1}: {ref_ans}\n"

    num_references_int = len(reference_answer_list)
    num_references_en = INT_TO_EN[num_references_int]

    # Generate the prompt
    prompt = (
        LINKAGE_PROMPT_TEMPLATE.replace("#ground", reference_answer_list_str)
        .replace("#candidate", candidate_answer)
        .replace("#question", question)
        .replace("{#num_references_int}", str(num_references_int))
        .replace("{#num_references_en}", num_references_en)
    )

    if use_gpt:
        formatted_chat_template = [
            {"role": "system", "content": DEFAULT_SYS_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return formatted_chat_template
    else:
        return prompt


# def linkage_run_gpt(
#     question: str,
#     reference_answers: List[str],
#     candidate_answer: str,
#     scorer_model_name: str
# ) -> int:
#     """
#     Uses GPT-4 to rank a candidate answer against a list of reference answers.

#     Parameters:
#     - question (str): The user's question.
#     - reference_answers (list of str): List of reference answers in descending order of quality. (e.g., [best_answer, good_answer, average_answer, poor_answer])
#     - candidate_answer (str): The answer to be ranked.
#     - scorer_model_name (str): The name of the GPT model to use for scoring.

#     Returns:
#     - rank (int): The ranking of the candidate answer.
#     """

#     # Generate the prompt using the make_prompt function
#     linkage_prompt = _make_linkage_prompt(question, reference_answers, candidate_answer)

#     # Call the OpenAI API
#     try:
#         response = OPENAI_CLIENT.chat.completions.create(
#             model=scorer_model_name,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": DEFAULT_SYS_PROMPT,
#                 },
#                 {
#                     "role": "user",
#                     "content": linkage_prompt,
#                 }
#             ],
#             temperature=LLM_TEMPERATURE,
#             top_p=LLM_TOP_P,
#             max_tokens=LLM_MAX_TOKENS
#         )
#         ranking_str = response.choices[0].message.content.strip()

#         # Extract the ranking from the response
#         rank = int(RANK_PATTERN.search(ranking_str).group(1))
#         assert rank > 0, f"Rank is not positive: {rank}"
#         return rank

#     except OPENAI_CLIENT.OpenAIError as e:
#         print(f"Error calling OpenAI API: {e}")
#         return None


# def linkage_run_vllm(
#     question: str,
#     reference_answers: List[str],
#     candidate_answer: str,
#     scorer_model_name: str
# ) -> int:
#     """
#     Uses Open LLM in the Hugging Face to rank a candidate answer against a list of reference answers.

#     Parameters:
#     - question (str): The user's question.
#     - reference_answers (list of str): List of reference answers in descending order of quality. (e.g., [best_answer, good_answer, average_answer, poor_answer])
#     - candidate_answer (str): The answer to be ranked.
#     - scorer_model_name (str): The name of the Hugging Face model to use for scoring.

#     Returns:
#     - rank (int): The ranking of the candidate answer.
#     """


def calculate_normalized_rank(rank: int, num_references: int) -> float:
    """
    Calculate the normalized rank from the raw rank and the number of reference answers.

    Args:
    - rank (int): The raw rank.
    - num_references (int): The number of reference answers.

    Returns:
    - normalized_rank (float): The normalized rank.
    """

    normalized_rank = (rank - 1) / (num_references)
    return normalized_rank


def make_linkage_prompt_batch(
    input_instances: List[Dict[str, Any]],
    use_gpt: bool,
    num_references: int = 4,
    return_instances: bool = False,
):
    """
    Make prompt batch for LINKAGE method.

    Args:
    - input_instances (list of dictionaries): The list of input instances.
    - use_gpt (bool): Whether to use GPT.
    - num_references (int): The number of reference answers.
    - return_reference_answers (bool): Whether to return the reference answers.

    Returns:
    - prompt_batch (list of str or list of list of dictionaries): The batch of prompts.
    - num_references_list (list of int): The list of number of reference answers.
    - input_instances (list of dictionaries): The list of input instances.
    """

    prompt_batch = []
    num_references_list = []

    for instance in input_instances:
        question = instance["question_text"]
        candidate_answer = instance["output"]

        answer_list = instance["answer_list"]
        answer_label = instance["answer_label"]

        grouped_answers = {label: [] for label in reversed(range(0, num_references))}
        for answer, label in zip(answer_list, answer_label):
            grouped_answers[label].append(answer)

        reference_answers = []
        for label in grouped_answers:
            if len(grouped_answers[label]) == 0:
                continue
            else:
                reference_answers.append(random.choice(grouped_answers[label]))

        instance["reference_answers"] = reference_answers
        num_references_list.append(len(reference_answers))

        prompt = _make_linkage_prompt(
            question, reference_answers, candidate_answer, use_gpt
        )
        prompt_batch.append(prompt)

    if return_instances:
        return prompt_batch, num_references_list, input_instances
    else:
        return prompt_batch, num_references_list


def make_batch_request_dict(
    model_name: str,
    custom_id_prefix: str,
    question_id_list: List[str],
    prompt_batch: List[List[Dict[str, str]]],
    return_input_token_length: bool = True,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
    max_tokens=LLM_MAX_TOKENS,
) -> List[Dict[str, Any]]:

    batch_request_dict = []
    total_input_token_length = 0

    for question_id, formatted_chat_template in zip(question_id_list, prompt_batch):

        if return_input_token_length:
            request_dict, input_token_length = make_single_request_dict(
                model_name=model_name,
                custom_id_prefix=custom_id_prefix,
                question_id=question_id,
                formatted_chat_template=formatted_chat_template,
                return_input_token_length=return_input_token_length,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            total_input_token_length += input_token_length
        else:
            request_dict = make_single_request_dict(
                model_name=model_name,
                custom_id_prefix=custom_id_prefix,
                question_id=question_id,
                formatted_chat_template=formatted_chat_template,
                return_input_token_length=return_input_token_length,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        batch_request_dict.append(request_dict)

    if return_input_token_length:
        return batch_request_dict, total_input_token_length
    else:
        return batch_request_dict


def linkage_eval_batch(
    scorer_model_name: str,
    prompt_batch: Union[List[str], List[List[Dict[str, str]]]],
    use_gpt: bool,
    openai_client: Optional[Any] = None,
    llm: Optional[LLM] = None,
    sampling_params: Optional[SamplingParams] = None,
) -> List[int]:
    """
    Evaluates a batch of prompts using the LINKAGE method.

    Args:
    - scorer_model_name (str): The name of the scoring model to use.
    - prompt_batch (list of str or list of list of dictionaries): The batch of prompts to evaluate.
    - num_references_list (list of int): The list of number of reference answers for each prompt.
    - use_gpt (bool): Whether to use GPT.
    - openai_client (OpenAI): The OpenAI client.
    - llm (LLM): The LLM model of vLLM to use.
    - sampling_params (SamplingParams): The sampling parameters for the LLM.
    - device (str): The device to use for the LLM.

    Returns:
    - ranks (list of int): The list of ranks for each prompt.
    """

    ranks = []

    if use_gpt:
        for idx, formatted_chat_template in enumerate(
            tqdm(prompt_batch, desc=f"Evaluating prompts using {scorer_model_name}")
        ):
            try:
                response = openai_client.chat.completions.create(
                    model=scorer_model_name,
                    messages=formatted_chat_template,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    max_tokens=LLM_MAX_TOKENS,
                )
                ranking_str = response.choices[0].message.content.strip()

                # Extract the ranking from the response
                try:
                    rank = int(RANK_PATTERN.search(ranking_str).group(1))
                    assert rank > 0, f"Rank is not positive: {rank}"
                except Exception as e:
                    max_trial = 5
                    trial = 0
                    is_success = False

                    while trial < max_trial:

                        try:
                            response = openai_client.chat.completions.create(
                                model=scorer_model_name,
                                messages=formatted_chat_template,
                                temperature=LLM_TEMPERATURE,
                                top_p=LLM_TOP_P,
                                max_tokens=LLM_MAX_TOKENS,
                            )
                        except openai_client.OpenAIError as e:
                            logging.error(
                                "Error in generating model output for prompt: %s. Error: %s",
                                formatted_chat_template,
                                e,
                            )
                            trial += 1
                            continue
                        ranking_str = response.choices[0].message.content.strip()
                        try:
                            rank = int(RANK_PATTERN.search(ranking_str).group(1))
                            assert rank > 0, f"Rank is not positive: {rank}"
                            is_success = True
                            break
                        except AttributeError:
                            print(
                                f"[{trial}/{max_trial}] Failed to extract rank from output: {ranking_str}"
                            )
                            trial += 1

                    if is_success is False:
                        rank = ranking_str

            except openai_client.OpenAIError as e:
                logging.error(
                    "Error in generating model output for prompt: %s. Error: %s",
                    formatted_chat_template,
                    e,
                )
                rank = None

            ranks.append(rank)
    else:
        vllm_output_list = llm.generate(prompt_batch, sampling_params)
        output_list = [output.outputs[0].text for output in vllm_output_list]
        for idx, output in enumerate(output_list):
            try:
                rank = int(RANK_PATTERN.search(output).group(1))
                assert rank > 0, f"Rank is not positive: {rank}"
            except Exception as e:
                max_trial = 100
                trial = 0
                is_success = False

                while trial < max_trial:
                    vllm_output = llm.generate(prompt_batch[idx], sampling_params)
                    output = vllm_output[0].outputs[0].text
                    try:
                        rank = int(RANK_PATTERN.search(output).group(1))
                        assert rank > 0, f"Rank is not positive: {rank}"
                        is_success = True
                        break
                    except Exception as e:
                        print(
                            f"[{trial}/{max_trial}] Failed to extract rank from output: {output}, {e}"
                        )
                        trial += 1

                if is_success is False:
                    rank = output

            ranks.append(rank)

    return ranks


def openai_batch_api_run(args):
    assert args.use_openai_batch_api, "OpenAI Batch API is not enabled."
    assert (
        args.scorer_model_name == "gpt-4o-mini-2024-07-18"
    ), "OpenAI Batch API is only available for GPT-4o-mini-2024-07-18"

    USE_GPT = True
    SCORER_MODEL_ALIAS = "gpt-4o-mini"
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    OPENAI_CLIENT = OpenAI(api_key=openai_api_key)

    batch_file_directory = os.path.join(
        "evaluation",
        "LINKAGE_results",
        "BatchAPI",
        args.model_alias_to_evaluate,
    )
    if not os.path.exists(batch_file_directory):
        os.makedirs(batch_file_directory)
    batch_info_dict_filepath = os.path.join(batch_file_directory, "batch_info.json")

    output_directory = os.path.join(
        "evaluation", "LINKAGE_results", args.model_alias_to_evaluate
    )

    if args.openai_batch_api_mode.lower() == "create":
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        input_directory = os.path.join(
            "experiments", "results", args.model_alias_to_evaluate
        )
        assert os.path.exists(
            input_directory
        ), f"Directory {input_directory} does not exist."

        file_list_to_evaluate = os.listdir(input_directory)
        print(f"=============Files to evaluate: {file_list_to_evaluate}")

        expected_total_input_price = 0
        expected_total_output_price = 0
        batch_info_dict = {}
        for file in file_list_to_evaluate:
            input_filepath = os.path.join(input_directory, file)
            assert os.path.isfile(input_filepath) and file.endswith(
                ".jsonl"
            ), f"File {input_filepath} is not a JSONL file."
            base_filename = os.path.splitext(file)[0]
            batch_info_dict[base_filename] = {}

            print(f"Reading {input_filepath}")
            input_instances = read_jsonl(input_filepath)

            question_id_list = [
                str(instance["question_id"]) for instance in input_instances
            ]

            prompt_batch, num_references_list, instances = make_linkage_prompt_batch(
                input_instances,
                USE_GPT,
                num_references=args.num_references,
                return_instances=True,
            )
            input_prompt_list = [
                formatted_chat_template[1]["content"]
                for formatted_chat_template in prompt_batch
            ]

            # (0-1) Make batch request dictionary
            batch_request_dict, total_input_token_length = make_batch_request_dict(
                model_name=args.scorer_model_name,
                custom_id_prefix=args.custom_id_prefix,
                question_id_list=question_id_list,
                prompt_batch=prompt_batch,
                return_input_token_length=True,
            )

            # (0-2) Write batch input file
            batch_input_filepath = os.path.join(
                batch_file_directory, f"{base_filename}_batch_input.jsonl"
            )
            write_jsonl(batch_request_dict, batch_input_filepath)

            # (0-3) Write misc. data
            misc_instances = []
            assert len(question_id_list) == len(num_references_list), "Length mismatch"
            assert len(num_references_list) == len(input_prompt_list), "Length mismatch"
            for idx in range(len(question_id_list)):
                misc_instance = {}
                misc_instance["question_id"] = question_id_list[idx]
                misc_instance["input_prompt"] = input_prompt_list[idx]
                misc_instance["num_references"] = num_references_list[idx]
                misc_instances.append(misc_instance)

            write_jsonl(
                misc_instances,
                os.path.join(
                    batch_file_directory, f"{base_filename}_misc.jsonl"
                ),
            )

            write_jsonl(
                instances,
                os.path.join(batch_file_directory, f"{base_filename}_instances.jsonl"),
            )

            # (0-4) Upload batch input file
            batch_input_file = OPENAI_CLIENT.files.create(
                file=open(batch_input_filepath, "rb"), purpose="batch"
            )
            batch_input_file_id = batch_input_file.id
            batch_info_dict[base_filename]["batch_input_file_id"] = batch_input_file_id

            # (1-1) Create batch [COST!!!]
            batch_obj = OPENAI_CLIENT.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "model": args.scorer_model_name,
                    "file": file,
                    "description": f"LINKAGE evaluation for {args.model_alias_to_evaluate}",
                    "num_references": str(args.num_references),
                },
            )

            batch_info_dict[base_filename]["batch_id"] = batch_obj.id
            batch_info_dict[base_filename]["batch_status"] = batch_obj.status

            # (1-2) Calculate expected total price
            expected_total_input_price += (
                total_input_token_length
                * PRICING_PER_INPUT_TOKEN[args.scorer_model_name]
            )
            expected_total_output_price += (
                LLM_MAX_TOKENS
                * len(question_id_list)
                * PRICING_PER_OUTPUT_TOKEN[args.scorer_model_name]
            )
            batch_info_dict[base_filename][
                "expected_input_price"
            ] = expected_total_input_price
            batch_info_dict[base_filename][
                "expected_output_price"
            ] = expected_total_output_price

        batch_info_dict["expected_total_input_price"] = expected_total_input_price
        batch_info_dict["expected_total_output_price"] = expected_total_output_price

        # Write batch info
        write_json(batch_info_dict, batch_info_dict_filepath)

    if args.openai_batch_api_mode.lower() == "retrieve":
        assert os.path.exists(
            output_directory
        ), f"Directory {output_directory} does not exist."
        
        assert os.path.exists(
            batch_info_dict_filepath
        ), f"Batch info file {batch_info_dict_filepath} does not exist."
        batch_info_dict = read_json(batch_info_dict_filepath)

        for base_filename in batch_info_dict:
            if base_filename in ["expected_total_input_price", "expected_total_output_price"]:
                continue
            batch_id = batch_info_dict[base_filename]["batch_id"]
            if batch_info_dict[base_filename].get("batch_status") == "completed":
                continue

            print(f"Retrieving batch {batch_id}")

            # (2) Retrieve batch
            batch_obj = OPENAI_CLIENT.batches.retrieve(batch_id)
            status = batch_obj.status
            metadata = batch_obj.metadata

            assert (
                metadata["model"] == args.scorer_model_name
            ), f"Model mismatch: {metadata['model']} vs {args.scorer_model_name}"
            assert (
                metadata["description"]
                == f"LINKAGE evaluation for {args.model_alias_to_evaluate}"
            ), f"Description mismatch: {metadata['description']} vs LINKAGE evaluation for {args.model_alias_to_evaluate}"
            assert (
                int(metadata["num_references"]) == args.num_references
            ), f"Num references mismatch: {metadata['num_references']} vs {args.num_references}"
            file = metadata["file"]
            base_filename = os.path.splitext(file)[0]

            print(f"Batch status for {base_filename}: {status}")

            if status == "completed":
                print(f"Result of {base_filename} ({batch_id}) will be analyzed now.")

                # (3-1) Download batch output
                batch_output_file_id = batch_obj.output_file_id
                batch_info_dict[base_filename][
                    "batch_output_file_id"
                ] = batch_output_file_id
                file_response = OPENAI_CLIENT.files.content(batch_output_file_id)

                batch_str_list = file_response.text.split("\n")[:-1]

                # (3-2) Parse batch output
                batch_output_instances = []
                for batch_str in tqdm(batch_str_list):
                    batch_output_instance = json.loads(batch_str)
                    batch_output_instances.append(batch_output_instance)

                # (3-3) Write batch output
                batch_output_filepath = os.path.join(
                    batch_file_directory, f"{base_filename}_batch_output.jsonl"
                )
                write_jsonl(batch_output_instances, batch_output_filepath)

                # (4-0) Read misc. data

                misc_instances = read_jsonl(
                    os.path.join(
                        batch_file_directory, f"{base_filename}_misc.jsonl"
                    )
                )

                question_id_list = [instance["question_id"] for instance in misc_instances]
                num_references_list = [instance["num_references"] for instance in misc_instances] 
                input_prompt_list = [instance["input_prompt"] for instance in misc_instances]

                instances = read_jsonl(
                    os.path.join(
                        batch_file_directory, f"{base_filename}_instances.jsonl"
                    )
                )

                # (4-1) Analyze batch output
                assert len(batch_output_instances) == len(
                    question_id_list
                ), f"Length mismatch: batch_output_instances ({len(batch_output_instances)}) vs question_id_list ({len(question_id_list)})"
                assert len(question_id_list) == len(
                    input_prompt_list
                ), f"Length mismatch: question_id_list ({len(question_id_list)}) vs input_prompt_list ({len(input_prompt_list)})"

                output_rank_list = []
                for idx in range(len(batch_output_instances)):
                    question_id = question_id_list[idx]
                    batch_output_instance = batch_output_instances[idx]
                    custom_id = batch_output_instance["response"]["body"]["id"]

                    # assert (
                        # f"{args.custom_id_prefix}-{question_id}" == custom_id
                    # ), f"Custom ID mismatch: {args.custom_id_prefix}-{question_id} vs {custom_id}"

                    ranking_str = batch_output_instance["response"]["body"]["choices"][
                        0
                    ]["message"]["content"].strip()

                    # Extract the ranking from the response
                    try:
                        rank = int(RANK_PATTERN.search(ranking_str).group(1))
                        assert rank > 0, f"Rank is not positive: {rank}"
                    except Exception as e:
                        max_trial = 5
                        trial = 0
                        is_success = False

                        # COST!!! - Retry
                        while trial < max_trial:
                            formatted_chat_template = [
                                {"role": "system", "content": DEFAULT_SYS_PROMPT},
                                {"role": "user", "content": input_prompt_list[idx]},
                            ]

                            try:
                                response = OPENAI_CLIENT.chat.completions.create(
                                    model=args.scorer_model_name,
                                    messages=formatted_chat_template,
                                    temperature=LLM_TEMPERATURE,
                                    top_p=LLM_TOP_P,
                                    max_tokens=LLM_MAX_TOKENS,
                                )
                            except OPENAI_CLIENT.OpenAIError as e:
                                logging.error(
                                    "Error in generating model output for prompt: %s. Error: %s",
                                    formatted_chat_template,
                                    e,
                                )
                                trial += 1
                                continue

                            ranking_str = response.choices[0].message.content.strip()
                            try:
                                rank = int(RANK_PATTERN.search(ranking_str).group(1))
                                assert rank > 0, f"Rank is not positive: {rank}"
                                is_success = True
                                break
                            except Exception as e:
                                print(
                                    f"[{trial}/{max_trial}] Failed to extract rank from output: {ranking_str}, {e}"
                                )
                                trial += 1

                        if is_success is False:
                            rank = ranking_str

                    output_rank_list.append(rank)

                # (4-2) Write batch output
                NR_list = []
                output_instances = []

                for rank, num_references, instance in zip(
                    output_rank_list, num_references_list, instances
                ):
                    instance["LINKAGE_rank"] = rank
                    instance["LINKAGE_normalized_rank"] = calculate_normalized_rank(
                        rank, num_references
                    )
                    NR_list.append(instance["LINKAGE_normalized_rank"])
                    output_instances.append(instance)

                output_jsonl_filepath = os.path.join(
                    output_directory,
                    f"{base_filename}_LINKAGE_results_by_{SCORER_MODEL_ALIAS}.jsonl",
                )
                write_jsonl(output_instances, output_jsonl_filepath)

                # (4-3) Calculate scores
                MNR_score = sum(NR_list) / len(NR_list)
                print(f"MNR (Mean Normalized Rank) score of {file}: {MNR_score}")
                MNRP_score = 1 - MNR_score
                print(
                    f"MNRP (Mean Normalized Rank Position) score of {file}: {MNRP_score}"
                )

                output_scores_json_filepath = os.path.join(
                    output_directory, f"{base_filename}_scores.json"
                )
                if os.path.exists(output_scores_json_filepath):
                    scores_dict = read_json(output_scores_json_filepath)
                else:
                    scores_dict = {}

                if scores_dict.get(SCORER_MODEL_ALIAS) is None:
                    scores_dict[SCORER_MODEL_ALIAS] = {}
                scores_dict[SCORER_MODEL_ALIAS]["MNR"] = f"{MNR_score:.4f}"
                scores_dict[SCORER_MODEL_ALIAS]["MNRP"] = f"{MNRP_score:.4f}"

                write_json(scores_dict, output_scores_json_filepath)

                print(
                    f"Results written to {output_scores_json_filepath} and {output_jsonl_filepath}"
                )

                # (4-4) Update batch info
                batch_info_dict[base_filename][
                    "batch_status"
                ] = status  # Update status as "completed"

                # Write batch info
                write_json(batch_info_dict, batch_info_dict_filepath)

    if args.openai_batch_api_mode.lower() == "cancel":
        batch_obj = OPENAI_CLIENT.batches.retrieve(
            args.openai_batch_api_batch_id_to_cancel
        )
        status = batch_obj.status
        print(f"Batch status for {args.openai_batch_api_batch_id_to_cancel}: {status}")

    if args.openai_batch_api_mode.lower() == "list":
        batch_list = OPENAI_CLIENT.batches.list()
        print("Batch list:")
        print(batch_list)


def main(args):
    OPENAI_CLIENT = None
    llm = None
    sampling_params = None

    if args.scorer_model_name == "gpt-4o-mini-2024-07-18":
        USE_GPT = True
        SCORER_MODEL_ALIAS = "gpt-4o-mini"
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        OPENAI_CLIENT = OpenAI(api_key=openai_api_key)
    elif args.scorer_model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        USE_GPT = False
        SCORER_MODEL_ALIAS = "mistral-7b-ins"

        device = "cuda"
        sampling_params = SamplingParams(
            temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, max_tokens=LLM_MAX_TOKENS
        )
        llm = LLM(
            model=args.scorer_model_name,
            device=device,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )

    output_directory = os.path.join(
        "evaluation", "LINKAGE_results", args.model_alias_to_evaluate
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_directory = os.path.join(
        "experiments", "results", args.model_alias_to_evaluate
    )
    assert os.path.exists(
        input_directory
    ), f"Directory {input_directory} does not exist."
    file_list_to_evaluate = os.listdir(input_directory)
    print(f"=============Files to evaluate: {file_list_to_evaluate}")
    for file in file_list_to_evaluate:
        input_filepath = os.path.join(input_directory, file)
        assert os.path.isfile(input_filepath) and file.endswith(
            ".jsonl"
        ), f"File {input_filepath} is not a JSONL file."
        base_filename = os.path.splitext(file)[0]

        print(f"Reading {input_filepath}")
        input_instances = read_jsonl(input_filepath)

        prompt_batch, num_references_list, instances = make_linkage_prompt_batch(
            input_instances,
            USE_GPT,
            num_references=args.num_references,
            return_instances=True,
        )
        output_rank_list = linkage_eval_batch(
            args.scorer_model_name,
            prompt_batch,
            USE_GPT,
            OPENAI_CLIENT,
            llm,
            sampling_params,
        )

        NR_list = []
        output_instances = []
        for rank, num_references, instance in zip(
            output_rank_list, num_references_list, instances
        ):
            instance["LINKAGE_rank"] = rank
            instance["LINKAGE_normalized_rank"] = calculate_normalized_rank(
                rank, num_references
            )
            NR_list.append(instance["LINKAGE_normalized_rank"])
            output_instances.append(instance)

        output_jsonl_filepath = os.path.join(
            output_directory,
            f"{base_filename}_LINKAGE_results_by_{SCORER_MODEL_ALIAS}.jsonl",
        )
        write_jsonl(output_instances, output_jsonl_filepath)

        MNR_score = sum(NR_list) / len(NR_list)
        print(f"MNR (Mean Normalized Rank) score of {file}: {MNR_score}")
        MNRP_score = 1 - MNR_score
        print(f"MNRP (Mean Normalized Rank Position) score of {file}: {MNRP_score}")

        output_scores_json_filepath = os.path.join(
            output_directory, f"{base_filename}_scores.json"
        )
        if os.path.exists(output_scores_json_filepath):
            scores_dict = read_json(output_scores_json_filepath)
        else:
            scores_dict = {}

        if scores_dict.get(SCORER_MODEL_ALIAS) is None:
            scores_dict[SCORER_MODEL_ALIAS] = {}
        scores_dict[SCORER_MODEL_ALIAS]["MNR"] = f"{MNR_score:.4f}"
        scores_dict[SCORER_MODEL_ALIAS]["MNRP"] = f"{MNRP_score:.4f}"

        write_json(scores_dict, output_scores_json_filepath)

        print(
            f"Results written to {output_scores_json_filepath} and {output_jsonl_filepath}"
        )


if __name__ == "__main__":
    set_seed(args.random_seed)

    if args.use_openai_batch_api:
        assert (
            args.scorer_model_name == "gpt-4o-mini-2024-07-18"
        ), "OpenAI Batch API is only available for GPT-4o-mini-2024-07-18"
        openai_batch_api_run(args)

    else:
        main(args)

    # calculate_expected_API_price()

    """
    # ------------- example startpoint-------------------------
    # non-factoid question
    question = "What is wifi vs bluetooth ?"
    
    # Reference Answers 
    # (4점 만점) Best Answer: 4점/4점, Good Answer: 3점/4점, Average Answer: 2점/4점, Poor Answer: 1점/4점
    best_answer = "Wi-Fi and Bluetooth are to some extent complementary in their applications andusage. Wi-Fi isusually access point-centered, with an asymmetrical client-server connection with all traffic routed through the access point, while Bluetooth is usually symmetrical, between two Bluetooth devices."
    good_answer = "Bluetooth vs. WiFi - Range: Maximum range for Bluetooth based wireless connections is 30m while for Wi-Fi, it can extend well upto 100m. In Wi-Fi, range depends on the version of Wi-Fi protocol applied and addition of antennas in the communication system while no such concerns of range or extra antenna are much known in Bluetooth. . Bluetooth vs. WiFi - Devices Connected: In Bluetooth, upto 7 devices can be connected to each other (piconet) while in Wi-Fi, the maximum connections depend on Wi-Fi router which can accommodate 1 to several communicating devices at a time."
    average_answer = "Bluetooth and WiFi are different standards for wireless communication. Bluetooth technology is useful when transferring information between two or more devices that are near each other when speed is not anissue,such as telephones, printers, modems and headsets."
    poor_answer = "Headphones use over 90% of available Bluetooth bandwidth. If you initiate anyother Bluetooth activity (view devices in range, or try to use any other Bluetooth services), the music may play intermittently, skip, or the headphone's synchronization with the audio source may disconnect."
    reference_answer_list = [best_answer, good_answer, average_answer, poor_answer]
    
    # LINKAGE 방식으로 평가하고자 하는 답변
    # 여기서는 우리 RAG 방법론으로 생성한 답변
    candidate_answer_bad = "Learn about Bluetooth and Wi-Fi for your Apple Watch, and why you should use both. To enjoy every feature on your Apple Watch, you need to turn on Wi-Fi and Bluetooth on your paired iPhone. Swipe up on your iPhone to open Control Center."
    candidate_answer_good = "You can also share a smart phone mobile data connection withother devices via the wireless Bluetooth radio. This is knownas a Bluetooth personal are a network, or PAN. Devices that include Bluetooth radios canc onnect to the smartphone via Bluetooth and access the Internet through it."
    # ------------- example endpoint-------------------------

    rank = linkage_run_gpt(question, reference_answer_list, candidate_answer_good)

    # 반복 평가 3회 결과 보자
    for i in range(3):
        print(i,"회")
        rank = linkage_run_gpt(question, reference_answer_list, candidate_answer_good)
        print(rank)
        print()
    """
