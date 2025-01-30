import json
from typing import List, Dict, Tuple, Union, Any

import tiktoken
from config import LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS


PRICING_PER_INPUT_TOKEN = {
    "gpt-4o": 1.25 / 1000000,
    "gpt-4o-2024-11-20": 1.25 / 1000000,
    "gpt-4o-2024-08-06": 1.25 / 1000000,
    "gpt-4o-2024-05-13": 2.50 / 1000000,
    "gpt-4o-mini": 0.075 / 1000000,
    "gpt-4o-mini-2024-07-18": 2.50 / 1000000,
}

PRICING_PER_OUTPUT_TOKEN = {
    "gpt-4o": 5 / 1000000,
    "gpt-4o-2024-11-20": 5 / 1000000,
    "gpt-4o-2024-08-06": 5 / 1000000,
    "gpt-4o-2024-05-13": 7.5 / 1000000,
    "gpt-4o-mini": 0.3 / 1000000,
    "gpt-4o-mini-2024-07-18": 0.3 / 1000000,
}


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


# def write_text(text: str, file_path: str):
#     with open(file_path, "w") as file:
#         file.write(text)


# def read_list(file_path: str) -> List:
#     with open(file_path, "r") as file:
#         instance = [line.strip() for line in file.readlines() if line.strip()]
#     return instance


# def write_list(instance: List, file_path: str):
#     with open(file_path, "w") as file:
#         for i in instance:
#             file.write(str(i) + "\n")


def make_single_request_dict(
    model_name: str,
    custom_id_prefix: str,
    question_id,
    formatted_chat_template: List[Dict],
    return_input_token_length: bool = False,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
    max_tokens=LLM_MAX_TOKENS,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
    """
    Create a single request dictionary for the given model and formatted chat template.

    Args:
    - model_name (str): The name of the model to use for the request.
    - custom_id_prefix (str):
        The custom id prefix to use for the request.
        It is needed to generate a unique `custom_id`
            based on how you configured the `formatted_chat_template`,
            because the `formatted_chat_template` can be configured differently, even for the same question.
    - question_id (str):
        The unique identifier for the question.
        It is required to be unique for each question.
    - formatted_chat_template (List[Dict]):
        The formatted chat template to use for the request.
            e.g.,
            ```
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
            ```
    - return_input_token_length (bool): Whether to return the input token length for the request.
    - temperature (float): The temperature to use for the request.
    - top_p (float): The top_p to use for the request.
    - max_tokens (int): The max_tokens to use for the request.

    Returns:
    - request_dict (Dict[str, Any]): The request dictionary for the given model and formatted chat template.
    - [Optional] input_token_length (int):
        The input token length for the single request.
        It is needed to calculate the total pricing of OpenAI Batch API according to the input token length.

    """
    request_dict = {
        "custom_id": f"{custom_id_prefix}-{question_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": formatted_chat_template,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    }

    if return_input_token_length:
        encoding = tiktoken.encoding_for_model(model_name)

        input_token_length = 0
        for chat in formatted_chat_template:
            input_token_length += len(encoding.encode(chat["content"]))

        return request_dict, input_token_length

    else:
        return request_dict
