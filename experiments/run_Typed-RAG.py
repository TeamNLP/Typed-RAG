import argparse
import ast
import logging
import os
import re
import sys
import json
from typing import Union, Any, Dict, List, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# sys.path.remove(parent_dir)

import dotenv
import numpy as np
import openai
import torch
from classifier.model import RobertaNFQAClassification
from classifier.lib import get_nfqa_category_prediction
from config import (
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    VLLM_GPU_MEMORY_UTILIZATION,
)
from langchain_huggingface import HuggingFaceEmbeddings
from lib import read_jsonl, write_jsonl
from openai import OpenAI
from prompt_templates import DEFAULT_SYS_PROMPT, RAG_SYS_PROMPT, RAG_PROMPT_TEMPLATE
from prompt_templates import (
    QUERY_ANALYST_SYS_PROMPT,
    DEBATE_MEDIATOR_SYS_PROMPT,
    AGGREGATOR_SYS_PROMPT,
)
from prompt_templates import DEBATE_SUBQUERY_GENERATION_PROMPT, DEBATE_MEDIATOR_PROMPT
from prompt_templates import EXPERIENCE_KEYWORD_EXTRACTION_PROMPT
from prompt_templates import (
    COMPARISON_KEYWORD_EXTRACTION_PROMPT,
    COMPARISON_GENERATOR_PROMPT,
)
from prompt_templates import REASON_QUERY_DECOMPOSITION_PROMPT
from prompt_templates import INSTRUCTION_QUERY_DECOMPOSITION_PROMPT
from prompt_templates import AGGREGATION_PROMPT
from retriever.bing_search import BingSearchRetriever
from retriever.bm25 import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEBUG_QUERY = "United States"

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(text: str) -> List[str]:
    p = re.compile("\[.*\]")
    result = p.findall(text)

    if len(result) > 1:
        list_str = result[0]
        try:
            output_list = ast.literal_eval(list_str)
            for s in output_list:
                assert isinstance(s, str), "Invalid list element."
        except Exception as e:
            logging.error("Error in converting string to list: %s", e)
            return []
    else:
        return []


def str2dict(text: str) -> Dict[str, Any]:
    p = re.compile("\{.*\}")
    result = p.findall(text)

    if len(result) > 1:
        dict_str = result[0]
        try:
            output_dict = ast.literal_eval(dict_str)
            for k, v in output_dict.items():
                assert isinstance(k, str) and isinstance(v, (str, int, float, bool)), "Invalid dictionary key-value pair."
        except Exception as e:
            logging.error("Error in converting string to dictionary: %s", e)
            return {}
    else:
        return {}


def generate_model_output(
    generator_model_path: str,
    prompt: Union[str, List[Dict[str, str]]],
    use_gpt: bool = False,
    system_prompt: Optional[str] = DEFAULT_SYS_PROMPT,
    openai_client: Optional[OpenAI] = None,
    llm: Optional[LLM] = None,
    sampling_params: Optional[SamplingParams] = None,
    device: str = "cuda",
) -> str:
    """
    Generate model output for the given prompt batch.

    Args:
    - generator_model_path: str - path to the model
    - prompt: Union[str, List[Dict[str, str]]] - prompt
    - use_gpt: bool - whether to use GPT
    - system_prompt: str - system prompt, default is DEFAULT_SYS_PROMPT
    - openai_client: Optional[OpenAI] - OpenAI client
    - llm: Optional[LLM] - LLM model of vLLM
    - sampling_params: Optional[SamplingParams] - sampling parameters
    - device: str - device to use

    Returns:
    - output_str: str - model output
    """

    if use_gpt:
        if isinstance(prompt, list):
            formatted_chat_template = prompt
        else:
            formatted_chat_template = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        try:
            response = openai_client.chat.completions.create(
                model=generator_model_path,
                messages=prompt,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
            )
            gpt_prediction = response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            logging.error(
                "Error in generating model output for prompt: %s. Error: %s",
                prompt,
                e,
            )
            gpt_prediction = ""
        return gpt_prediction
    else:
        if system_prompt != DEFAULT_SYS_PROMPT:
            prompt = f"{system_prompt}\n{prompt}"

        return llm.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].text


def generate_model_output_batch(
    generator_model_path: str,
    prompt_batch: Union[List[str], List[List[Dict[str, str]]]],
    use_gpt: bool = False,
    openai_client: Optional[OpenAI] = None,
    llm: Optional[LLM] = None,
    sampling_params: Optional[SamplingParams] = None,
    device: str = "cuda",
) -> List[str]:
    """
    Generate model output for the given prompt batch.

    Args:
    - generator_model_path: str - path to the model
    - prompt_batch: List[str] - list of prompts
    - openai_client: Optional[OpenAI] - OpenAI client
    - llm: Optional[LLM] - LLM model of vLLM
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
                    model=generator_model_path,
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
        vllm_output_list = llm.generate(prompt_batch, sampling_params)
        output_list = [output.outputs[0].text for output in vllm_output_list]

    return output_list


def rerank_passages(
    hf_embeddings: HuggingFaceEmbeddings,
    question_text: str,
    passages_list: List[str],
    top_k: int = 5,
    return_type: str = "list",
) -> List[str]:
    """
    Rerank passages for the given question text.

    Args:
    - hf_embeddings: HuggingFaceEmbeddings - HuggingFace embeddings
    - question_text: str - question text
    - passages_list: List[str] - list of passages
    - top_k: int - top k passages to return

    Returns:
    - reranked_passages_text or reranked_passages_list: Union[str, List[str]] - reranked passages text or reranked passages list
    """

    embedded_query = hf_embeddings.embed_query(question_text)
    embedded_documents = hf_embeddings.embed_documents(passages_list)

    embedded_query_np = np.array(embedded_query)
    embedded_query_reshaped = embedded_query_np.reshape(1, -1)
    try:
        cosine_similarities = cosine_similarity(embedded_documents, embedded_query_reshaped)
    except ValueError as e:
        print(e)
        print("length of embedded_documents:", len(embedded_documents))
        print("shape of embedded_query_reshaped:", embedded_query_reshaped.shape)
        print("passages_list:", passages_list)
        print("len(passages_list):", len(passages_list))
        print("question_text:", question_text)

        exit()


    top_k_indices = np.argsort(cosine_similarities, axis=0)[::-1][:top_k]
    # top_k_similarities = cosine_similarities[top_k_indices]
    reranked_passages_list = [
        passages_list[idx_array[0]] for idx_array in top_k_indices
    ]

    if return_type.lower() in ["text", "str", "string"]:
        reranked_passages_text = ""
        for passage in reranked_passages_list:
            reranked_passages_text += passage
            reranked_passages_text += "\n"
        reranked_passages_text = reranked_passages_text.strip()
        return reranked_passages_text
    elif return_type.lower() in ["list"]:
        return reranked_passages_list


def make_RAG_prompt(
    question_text: str,
    retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] = None,
    use_gpt: bool = False,
) -> Union[List[Dict[str, str]], str]:
    """
    Make RAG prompt for the given question text.

    Args:
    - question_text: str - question text
    - retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] - BM25 retriever
    - use_gpt: bool - whether to use GPT

    Returns:
    - formatted_chat_template or input_prompt: Union[List[Dict[str, str]], str] - formatted chat template or input prompt
    """
    references_text = retriever.retrieve(question_text, return_type="str")
    input_prompt = RAG_PROMPT_TEMPLATE.format(
        references=references_text, query=question_text
    )

    if use_gpt:  # Format prompt for OpenAI API if using GPT
        formatted_chat_template = [
            {"role": "system", "content": RAG_SYS_PROMPT},
            {"role": "user", "content": input_prompt},
        ]
        return formatted_chat_template
    else:
        input_prompt = f"{RAG_SYS_PROMPT}\n{input_prompt}"
        return input_prompt


def make_NFQA_prompt_batch(
    args: argparse.Namespace,
    input_instances: List[Dict[str, Any]],
    nfqa_model: RobertaNFQAClassification,
    nfqa_tokenizer: AutoTokenizer,
    openai_client: Optional[OpenAI] = None,
    llm: Optional[LLM] = None,
    sampling_params: Optional[SamplingParams] = None,
    retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] = None,
    hf_embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Union[List[List[Dict[str, str]]], List[str]]:
    """
    Make prompt batch for the given input instances.

    Args:
    - args: argparse.Namespace - arguments
    - input_instances: List[Dict[str, Any]] - list of input instances
    - nfqa_model: RobertaNFQAClassification - NFQA model
    - nfqa_tokenizer: AutoTokenizer - NFQA tokenizer
    - openai_client: Optional[OpenAI] - OpenAI client
    - llm: Optional[LLM] - LLM model of vLLM
    - sampling_params: Optional[SamplingParams] - sampling parameters
    - retriever: Optional[Union[BM25Retriever, BingSearchRetriever]] - retriever

    Returns:
    - prompt_batch: Union[List[List[Dict[str, str]]], List[str]] - prompt batch
    """

    prompt_batch = []

    for instance in tqdm(input_instances, desc="Making prompt batch for NFQA"):
        question_text = instance["question_text"]
        category_prediction = get_nfqa_category_prediction(
            nfqa_model, nfqa_tokenizer, question_text.strip(), device
        )

        if (
            category_prediction == "NOT-A-QUESTION"
            or category_prediction == "FACTOID"
            or category_prediction == "EVIDENCE-BASED"
        ):
            # START OF FACTOID, EVIDENCE-BASED, OR NOT-A-QUESTION
            prompt_batch.append(make_RAG_prompt(question_text, retriever, args.use_gpt))
            continue
            # END OF FACTOID, EVIDENCE-BASED, OR NOT-A-QUESTION
        elif category_prediction == "DEBATE":
            # START OF DEBATE
            # Generate sub-queries for debate-type questions

            def check_valid_output(output_str: str) -> bool:
                try:
                    output = json.loads(output_str.strip())
                    assert isinstance(output, dict), "Invalid output."
                    assert "debate_topic" in output, "Debate topic not found."
                    assert "dist_opinion" in output, "Distribution of opinions not found."
                    assert "sub-queries" in output, "Sub-queries not found."
                    assert (
                        isinstance(output["debate_topic"], str)
                        and isinstance(output["dist_opinion"], list)
                        and isinstance(output["sub-queries"], dict)
                    ), "Invalid sub-query generation output."
                    assert (
                        len(output["dist_opinion"]) >= 2 and len(output["dist_opinion"]) <= 5
                    ), "Invalid number of opinions."
                    assert (
                        len(output["dist_opinion"]) == len(output["sub-queries"])
                    ), "Number of opinions and sub-queries do not match."
                    for dist_op in output["dist_opinion"]:
                        assert isinstance(dist_op, str), "Invalid opinion."
                        assert dist_op in output["sub-queries"], "Opinion not found in sub-queries."
                        assert isinstance(output["sub-queries"][dist_op]), "Invalid sub-query."
                    return True
                except Exception as e:
                    return False

            for trial in range(40):
                subquery_generation_output_str = generate_model_output(
                    generator_model_path=args.generator_model_path,
                    prompt=DEBATE_SUBQUERY_GENERATION_PROMPT.format(query=question_text),
                    use_gpt=args.use_gpt,
                    system_prompt=QUERY_ANALYST_SYS_PROMPT,
                    openai_client=openai_client,
                    llm=llm,
                    sampling_params=sampling_params,
                    device=device,
                )

                if check_valid_output(subquery_generation_output_str):
                    break

            if not check_valid_output(subquery_generation_output_str):
                prompt_batch.append(
                    make_RAG_prompt(question_text, retriever, args.use_gpt)
                )
                continue

            subquery_generation_output = json.loads(subquery_generation_output_str)
            sub_queries = subquery_generation_output["sub-queries"]
            sub_opinions = {}
            for opinion, sub_query in sub_queries.items():
                input_prompt = RAG_PROMPT_TEMPLATE.format(
                    references=retriever.retrieve(sub_query, return_type="str"),
                    query=sub_query,
                )
                output_str = generate_model_output(
                    generator_model_path=args.generator_model_path,
                    prompt=input_prompt,
                    use_gpt=args.use_gpt,
                    system_prompt=RAG_SYS_PROMPT,
                    openai_client=openai_client,
                    llm=llm,
                    sampling_params=sampling_params,
                    device=device,
                )

                sub_opinions[opinion] = output_str

            dist_opinion = subquery_generation_output["dist_opinion"]
            participant_responses = ""
            for i in range(1, len(dist_opinion) + 1):
                opinion = dist_opinion[i - 1]
                participant_responses += f"Response {i}: {sub_opinions[opinion]} (Perspective: {sub_queries[opinion]})\n"

            debate_topic = subquery_generation_output["debate_topic"]
            input_prompt = DEBATE_MEDIATOR_PROMPT.format(
                debate_topic=debate_topic, responses=participant_responses
            )

            if args.use_gpt:  # Format prompt for OpenAI API if using GPT
                formatted_chat_template = [
                    {"role": "system", "content": DEBATE_MEDIATOR_SYS_PROMPT},
                    {"role": "user", "content": input_prompt},
                ]
                prompt_batch.append(formatted_chat_template)
            else:
                input_prompt = f"{DEBATE_MEDIATOR_SYS_PROMPT}\n{input_prompt}"
                prompt_batch.append(input_prompt)
            continue
            # END OF DEBATE
        elif category_prediction == "INSTRUCTION" or category_prediction == "REASON":
            # START OF INSTRUCTION OR REASON
            if category_prediction == "INSTRUCTION":
                decomposition_prompt = INSTRUCTION_QUERY_DECOMPOSITION_PROMPT
            elif category_prediction == "REASON":
                decomposition_prompt = REASON_QUERY_DECOMPOSITION_PROMPT

            input_prompt = decomposition_prompt.format(query=question_text)
            sub_query_list_str = generate_model_output(
                generator_model_path=args.generator_model_path,
                prompt=input_prompt,
                use_gpt=args.use_gpt,
                system_prompt=QUERY_ANALYST_SYS_PROMPT,
                openai_client=openai_client,
                llm=llm,
                sampling_params=sampling_params,
                device=device,
            )

            try:
                sub_query_list = str2list(sub_query_list_str)

                if len(sub_query_list) == 0:
                    for trial in range(20):
                        sub_query_list_str = generate_model_output(
                            generator_model_path=args.generator_model_path,
                            prompt=input_prompt,
                            use_gpt=args.use_gpt,
                            system_prompt=QUERY_ANALYST_SYS_PROMPT,
                            openai_client=openai_client,
                            llm=llm,
                            sampling_params=sampling_params,
                            device=device,
                        )
                        sub_query_list = str2list(sub_query_list_str)
                        if len(sub_query_list) > 0:
                            break

                assert isinstance(sub_query_list, list), "Invalid sub-query list."
                assert len(sub_query_list) > 0, "Empty sub-query list."
            except Exception as e:
                prompt_batch.append(
                    make_RAG_prompt(question_text, retriever, args.use_gpt)
                )
                continue

            sub_answer_list = []
            for sub_query in sub_query_list:
                input_prompt = RAG_PROMPT_TEMPLATE.format(
                    references=retriever.retrieve(sub_query, return_type="str"),
                    query=sub_query,
                )
                sub_answer_str = generate_model_output(
                    generator_model_path=args.generator_model_path,
                    prompt=input_prompt,
                    use_gpt=args.use_gpt,
                    system_prompt=RAG_SYS_PROMPT,
                    openai_client=openai_client,
                    llm=llm,
                    sampling_params=sampling_params,
                    device=device,
                )
                sub_answer_list.append(sub_answer_str)

            try:
                assert len(sub_query_list) == len(
                    sub_answer_list
                ), "Number of sub-queries and sub-answers do not match."
            except Exception as e:
                prompt_batch.append(
                    make_RAG_prompt(question_text, retriever, args.use_gpt)
                )
                continue

            # Aggregate sub-answers to form the final answer
            qa_pairs_text = ""
            for i, sub_query in enumerate(sub_query_list):
                qa_pairs_text += f"Q: {sub_query}\nA: {sub_answer_list[i]}\n\n"

            input_prompt = AGGREGATION_PROMPT.format(
                original_question=question_text, qa_pairs_text=qa_pairs_text
            )

            if args.use_gpt:  # Format prompt for OpenAI API if using GPT
                formatted_chat_template = [
                    {"role": "system", "content": AGGREGATOR_SYS_PROMPT},
                    {"role": "user", "content": input_prompt},
                ]
                prompt_batch.append(formatted_chat_template)
            else:
                input_prompt = f"{AGGREGATOR_SYS_PROMPT}\n{input_prompt}"
                prompt_batch.append(input_prompt)
            continue
            # END OF INSTRUCTION OR REASON
        elif category_prediction == "EXPERIENCE":
            # START OF EXPERIENCE
            input_prompt = EXPERIENCE_KEYWORD_EXTRACTION_PROMPT.format(
                query=question_text
            )
            keywords_str = generate_model_output(
                generator_model_path=args.generator_model_path,
                prompt=input_prompt,
                use_gpt=args.use_gpt,
                system_prompt=QUERY_ANALYST_SYS_PROMPT,
                openai_client=openai_client,
                llm=llm,
                sampling_params=sampling_params,
                device=device,
            )

            references_list = retriever.retrieve(question_text, return_type="list")

            reranked_passages_str = rerank_passages(
                hf_embeddings,
                keywords_str,
                references_list,
                top_k=args.reranker_top_n,
                return_type="str",
            )

            input_prompt = RAG_PROMPT_TEMPLATE.format(
                references=reranked_passages_str, query=question_text
            )

            if args.use_gpt:  # Format prompt for OpenAI API if using GPT
                formatted_chat_template = [
                    {"role": "system", "content": RAG_SYS_PROMPT},
                    {"role": "user", "content": input_prompt},
                ]
                prompt_batch.append(formatted_chat_template)
            else:
                input_prompt = f"{RAG_SYS_PROMPT}\n{input_prompt}"
                prompt_batch.append(input_prompt)
            continue
            # END OF EXPERIENCE
        elif category_prediction == "COMPARISON":
            # START OF COMPARISON
            input_prompt = COMPARISON_KEYWORD_EXTRACTION_PROMPT.format(
                query=question_text
            )
            analysis_dict_str = generate_model_output(
                generator_model_path=args.generator_model_path,
                prompt=input_prompt,
                use_gpt=args.use_gpt,
                system_prompt=QUERY_ANALYST_SYS_PROMPT,
                openai_client=openai_client,
                llm=llm,
                sampling_params=sampling_params,
                device=device,
            )

            try:
                analysis_dict = str2dict(analysis_dict_str)

                if not analysis_dict:
                    for trial in range(20):
                        analysis_dict_str = generate_model_output(
                            generator_model_path=args.generator_model_path,
                            prompt=input_prompt,
                            use_gpt=args.use_gpt,
                            system_prompt=QUERY_ANALYST_SYS_PROMPT,
                            openai_client=openai_client,
                            llm=llm,
                            sampling_params=sampling_params,
                            device=device,
                        )
                        analysis_dict = str2dict(analysis_dict_str)
                        if analysis_dict:
                            break

                assert isinstance(analysis_dict, dict), "Invalid analysis dictionary."
                if isinstance(analysis_dict["is_compare"], str):
                    analysis_dict["is_compare"] = (
                        analysis_dict["is_compare"].lower() == "true"
                    )
                assert isinstance(
                    analysis_dict["is_compare"], bool
                ), "Invalid is_compare value."
                assert isinstance(
                    analysis_dict["compare_type"], str
                ), "Invalid compare_type value."
                if isinstance(analysis_dict["keywords_list"], str):
                    analysis_dict["keywords_list"] = str2list(
                        analysis_dict["keywords_list"]
                    )
                assert isinstance(
                    analysis_dict["keywords_list"], list
                ), "Invalid keywords_list value."
                assert len(analysis_dict["keywords_list"]) > 0, "Empty keywords_list."
                assert analysis_dict["is_compare"] == True, "Not a comparison question."
            except Exception as e:
                prompt_batch.append(
                    make_RAG_prompt(question_text, retriever, args.use_gpt)
                )
                continue

            keywords_list = analysis_dict["keywords_list"]

            references_list = []
            for keyword in keywords_list:
                references_list.extend(retriever.retrieve(keyword, return_type="list"))
            references_list = list(set(references_list))

            reranked_passages_str = rerank_passages(
                hf_embeddings,
                question_text,
                references_list,
                top_k=args.reranker_top_n,
                return_type="str",
            )

            input_prompt = COMPARISON_GENERATOR_PROMPT.format(
                query=question_text,
                comparison_type="",
                keywords_list=keywords_list,
                references=reranked_passages_str,
            )

            if args.use_gpt:  # Format prompt for OpenAI API if using GPT
                formatted_chat_template = [
                    {"role": "system", "content": RAG_SYS_PROMPT},
                    {"role": "user", "content": input_prompt},
                ]
                prompt_batch.append(formatted_chat_template)
            else:
                input_prompt = f"{RAG_SYS_PROMPT}\n{input_prompt}"
                prompt_batch.append(input_prompt)
            continue
            # END OF COMPARISON

    return prompt_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator_model_path",
        type=str,
        help="model path for the generator.",
        required=True,
        choices=(
            "gpt-4o-mini-2024-07-18",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
        ),
    )
    parser.add_argument("--use_gpt", type=str2bool, default=False)
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
    parser.add_argument("--reranker_top_n", type=int, default=5)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()


    nfqa_model = RobertaNFQAClassification.from_pretrained("Lurunchik/nf-cats").to(
        device
    )
    nfqa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=args.reranker_model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    openai_client = None
    llm = None
    sampling_params = None

    if args.generator_model_path.startswith("gpt"):
        if args.use_gpt is None:
            args.use_gpt = True
        else:
            assert args.use_gpt, "GPT model path provided but use_gpt is False."
        logging.getLogger("openai").setLevel(logging.ERROR)
        api_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)

        if args.generator_model_path == "gpt-4o-mini-2024-07-18":
            model_alias = "gpt-4o-mini"
    else:
        if args.use_gpt is None:
            args.use_gpt = False
        else:
            assert args.use_gpt == False, "LLM model path provided but use_gpt is True."
        
        llm = LLM(
            model=args.generator_model_path,
            device=device,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        
        sampling_params = SamplingParams(
            temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, max_tokens=LLM_MAX_TOKENS
        )

        if args.generator_model_path == "mistralai/Mistral-7B-Instruct-v0.2":
            model_alias = "mistral-7b-ins"
        elif args.generator_model_path == "meta-llama/Llama-3.2-3B-Instruct":
            model_alias = "llama-3.2-3b-ins"
        elif args.generator_model_path == "meta-llama/Llama-3.1-70B-Instruct":
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
        output_directory, f"Typed-RAG_prediction_{args.dataset_name}_{file_base_name}.jsonl"
    )  # jsonl file
    if args.debug:
        logging.debug("output file path: %s", output_filepath)
    output_instance = {}

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
            "Top %d retrieved documents: %s", args.retriever_top_n, str(passages_list)
        )

    prompt_batch = make_NFQA_prompt_batch(
        args,
        input_instances,
        nfqa_model,
        nfqa_tokenizer,
        openai_client,
        llm,
        sampling_params,
        retriever,
        hf_embeddings,
    )

    output_list = generate_model_output_batch(
        args.generator_model_path,
        prompt_batch,
        args.use_gpt,
        openai_client,
        llm,
        sampling_params,
        device,
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

    prompt_batch_output_directory = os.path.join("experiments", "prompt_batch", model_alias)
    if not os.path.exists(prompt_batch_output_directory):
        os.makedirs(prompt_batch_output_directory)

    prompt_batch_filepath = os.path.join(
        prompt_batch_output_directory, f"Typed-RAG_prediction_{args.dataset_name}_{file_base_name}.jsonl"
    )  # jsonl file
    prompt_batch_output_instances = []
    for idx in range(len(output_instances)):
        prompt_batch_output_instances.append(
            {
                "question_id": output_instances[idx]["question_id"],
                "question_text": output_instances[idx]["question_text"],
                "prompt_batch": prompt_batch[idx],
                "output": output_list[idx],
            }
        )
    write_jsonl(prompt_batch_output_instances, prompt_batch_filepath)

if __name__ == "__main__":
    main()
