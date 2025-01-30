import argparse
import logging
import os
import sys
from typing import Union, Any, Dict, List, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# sys.path.remove(parent_dir)

import dotenv
import openai
import torch
from config import (
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    VLLM_GPU_MEMORY_UTILIZATION,
)
from lib import read_jsonl, write_jsonl
from openai import OpenAI
from prompt_templates import DEFAULT_SYS_PROMPT
from prompt_templates import LLM_SYS_PROMPT, LLM_PROMPT_TEMPLATE
from prompt_templates import RAG_SYS_PROMPT, RAG_PROMPT_TEMPLATE
from retriever.bing_search import BingSearchRetriever
from retriever.bm25 import BM25Retriever
from tqdm import tqdm
from vllm import LLM, SamplingParams


DEBUG_QUERY = "What is the capital of South Korea?"

DATASET_TO_FILE_NAME = {
    "2wikimultihopqa": "annotated_odqa_nf_test.jsonl",
    "hotpotqa": "annotated_odqa_nf_test.jsonl",
    "musique": "annotated_odqa_nf_test.jsonl",
    "nq": "annotated_odqa_nf_test.jsonl",
    "squad": "annotated_odqa_nf_test.jsonl",
    "trivia": "annotated_odqa_nf_test.jsonl",
    "webglmqa": "test.jsonl",
    "antique": "test.jsonl",
    "trecdlnf": "test.jsonl",
}
DATASET_TO_RETRIEVER = {
    "2wikimultihopqa": "BM25Retriever",
    "hotpotqa": "BM25Retriever",
    "musique": "BM25Retriever",
    "nq": "BM25Retriever",
    "squad": "BM25Retriever",
    "trivia": "BM25Retriever",
    "webglmqa": "BingSearchRetriever",
    "antique": "BingSearchRetriever",
    "trecdlnf": "BingSearchRetriever",
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dotenv.load_dotenv()
logging.getLogger("httpx").setLevel(logging.ERROR)


def make_prompt_batch(
    prompt_template: str,
    system_prompt: str,
    input_instances: List[Dict[str, Any]],
    method: str,
    use_gpt: bool,
    retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] = None,
) -> Union[List[List[Dict[str, str]]], List[str]]:
    """
    Make prompt batch for the given input instances.

    Args:
    - prompt_template: str - prompt template
    - system_prompt: str - system prompt
    - input_instances: List[Dict[str, Any]] - list of input instances
    - method: str - method
    - use_gpt: bool - whether to use GPT
    - retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] - retriever

    Returns:
    - prompt_batch: Union[List[List[Dict[str, str]]], List[str]] - prompt batch
    """

    prompt_batch = []

    for instance in tqdm(input_instances, desc="Making prompt batch"):
        question_text = instance["question_text"]

        if method == "LLM":
            input_prompt = prompt_template.format(query=question_text)
        elif method == "RAG":
            references_text = retriever.retrieve(question_text, return_type="str")
            input_prompt = prompt_template.format(
                references=references_text, query=question_text
            )

        if use_gpt:  # Format prompt for OpenAI API if using GPT
            formatted_chat_template = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt},
            ]
            prompt_batch.append(formatted_chat_template)
        else:
            input_prompt = f"{system_prompt}\n{input_prompt}"
            prompt_batch.append(input_prompt)

    return prompt_batch


def generate_model_output_batch(
    model_path: str,
    prompt_batch: Union[List[str], List[List[Dict[str, str]]]],
    use_gpt: bool = False,
    openai_client: Optional[OpenAI] = None,
    sampling_params: Optional[SamplingParams] = None,
    device: str = "cuda",
) -> List[str]:
    """
    Generate model output for the given prompt batch.

    Args:
    - model_path: str - path to the model
    - prompt_batch: List[str] - list of prompts
    - openai_client: Optional[OpenAI] - OpenAI client
    - sampling_params: Optional[SamplingParams] - sampling parameters
    - device: str - device to use

    Returns:
    - output_list: List[str] - list of model outputs
    """

    output_list = []

    if use_gpt:
        for formatted_chat_template in tqdm(
            prompt_batch, desc="Generating model output using OpenAI GPT"
        ):
            try:
                response = openai_client.chat.completions.create(
                    model=model_path,
                    messages=formatted_chat_template,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    max_tokens=LLM_MAX_TOKENS,
                )
                gpt_prediction = response.choices[0].message.content.strip()
            except openai.error.OpenAIError as e:
                logging.error(
                    "Error in generating model output for prompt: %s. Error: %s",
                    formatted_chat_template,
                    e,
                )
                gpt_prediction = ""
            output_list.append(gpt_prediction)
    else:
        llm = LLM(
            model=model_path,
            device=device,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        vllm_output_list = llm.generate(prompt_batch, sampling_params)
        output_list = [output.outputs[0].text for output in vllm_output_list]

    return output_list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, help="method.", required=True, choices=("LLM", "RAG")
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path.",
        required=True,
        choices=(
            "gpt-4o-mini-2024-07-18",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset name.",
        required=True,
        choices=(
            "hotpotqa",
            "2wikimultihopqa",
            "musique",
            "nq",
            "trivia",
            "squad",
            "webglmqa",
            "antique",
            "trecdlnf",
        ),
    )
    parser.add_argument(
        "--file_name",
        type=str,
        help="file name. (e.g., `nf_test_300_filtered.jsonl`)",
        default=None,
    )
    parser.add_argument("--retriever_top_n", type=int, default=5)
    parser.add_argument(
        "--retriever_mode",
        type=str,
        help="Elasticsearch service mode (docker, executable, or existing)",
        default="docker",
        choices=["docker", "executable", "existing"],
    )
    parser.add_argument("--retriever_reindexing", type=str2bool, default=False)
    parser.add_argument(
        "--reranker_model_path",
        type=str,
        help="model path for the reranker.",
        required=False,
        default="BAAI/bge-reranker-large",
    )
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    openai_client = None
    sampling_params = None

    if args.model_path.startswith("gpt"):
        use_gpt = True
        logging.getLogger("openai").setLevel(logging.ERROR)
        api_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)

        if args.model_path == "gpt-4o-mini-2024-07-18":
            model_alias = "gpt-4o-mini"
    else:
        use_gpt = False

        if args.model_path == "mistralai/Mistral-7B-Instruct-v0.2":
            model_alias = "mistral-7b-ins"
        elif args.model_path == "meta-llama/Llama-3.2-3B-Instruct":
            model_alias = "llama-3.2-3b-ins"
        elif args.model_path == "meta-llama/Llama-3.1-70B-Instruct":
            model_alias = "llama-3.1-70b-ins"

    if args.file_name is None:
        if args.dataset_name not in DATASET_TO_FILE_NAME:
            logging.error("File name not found for dataset: %s", args.dataset_name)
            return
        args.file_name = DATASET_TO_FILE_NAME[args.dataset_name]

    input_directory = os.path.join(
        "data", "reference_list_construction", args.dataset_name
    )
    input_filepath = os.path.join(input_directory, args.file_name)
    if args.debug:
        logging.debug("input file path: %s", input_filepath)
    input_instances = read_jsonl(input_filepath)

    output_directory = os.path.join("experiments", "results", model_alias)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_base_name = os.path.splitext(args.file_name)[0]
    output_filepath = os.path.join(
        output_directory,
        f"{args.method}_prediction_{args.dataset_name}_{file_base_name}.jsonl",
    )  # jsonl file
    if args.debug:
        logging.debug("output file path: %s", output_filepath)

    if args.method == "LLM":
        system_prompt = LLM_SYS_PROMPT
        prompt_template = LLM_PROMPT_TEMPLATE

        prompt_batch = make_prompt_batch(
            prompt_template, system_prompt, input_instances, args.method, use_gpt
        )
    elif args.method == "RAG":
        system_prompt = RAG_SYS_PROMPT
        prompt_template = RAG_PROMPT_TEMPLATE

        if args.dataset_name not in DATASET_TO_RETRIEVER:
            logging.error("Retriever not found for dataset: %s", args.dataset_name)
            return
        elif DATASET_TO_RETRIEVER[args.dataset_name] == "BingSearchRetriever":
            retriever = BingSearchRetriever(
                top_n=args.retriever_top_n,
                reranker_model_path=args.reranker_model_path,
                dataset_name=args.dataset_name,
            )
        elif DATASET_TO_RETRIEVER[args.dataset_name] == "BM25Retriever":
            retriever = BM25Retriever(
                corpus_name="wiki",
                top_n=args.retriever_top_n,
                reindexing=args.retriever_reindexing,
                mode=args.retriever_mode,
            )

        if args.debug:
            query = DEBUG_QUERY
            passages_list = retriever.retrieve(query, return_type="list")
            logging.debug("Test Query for Retriever: %s", query)
            logging.debug(
                "Top %d retrieved documents: %s",
                args.retriever_top_n,
                str(passages_list),
            )

        prompt_batch = make_prompt_batch(
            prompt_template,
            system_prompt,
            input_instances,
            args.method,
            use_gpt,
            retriever,
        )

    output_list = generate_model_output_batch(
        args.model_path, prompt_batch, use_gpt, openai_client, sampling_params, device
    )
    assert len(input_instances) == len(
        output_list
    ), "Number of input instances and outputs do not match."

    output_instances = []
    for idx in range(len(input_instances)):
        question_id = input_instances[idx]["question_id"]
        question_text = input_instances[idx]["question_text"]
        output = output_list[idx]
        answer_list = input_instances[idx]["answer_list"]
        answer_label = input_instances[idx]["answer_label"]

        output_instances.append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "output": output,
                "answer_list": answer_list,
                "answer_label": answer_label,
            }
        )
    write_jsonl(output_instances, output_filepath)


if __name__ == "__main__":
    main()
