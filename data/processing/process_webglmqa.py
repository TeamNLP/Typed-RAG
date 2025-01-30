import os
import random

from config import PROCESSED_DATA, RAW_DATA, RANDOM_SEED
from lib import read_json, read_jsonl, write_jsonl


random.seed(RANDOM_SEED)


def sample_jsonl(data: list, sample_size: int = 50) -> list:
    sample_size = min(sample_size, len(data))
    sampled_data = random.sample(data, sample_size)
    
    return sampled_data


def transform_jsonl_data(data):
    """
    output jsonl format:
    {
        "question_id": 1, 
        "question_text": "how much of an effect will the religious freedom act in Indiana actually have on people's lives?", 
        "answer_objects": [
            {
                "number": "", 
                "date": {
                    "day": "", 
                    "month": "", 
                    "year": ""
                }, 
                "spans": ["Only a single answer"]
            }
        ],
        "contexts": [
            {
                "idx": 0, 
                "title": "", 
                "paragraph_text": "Reference text", 
                "is_supporting": false
            }, 
            {
                "idx": 1, 
                "title": "", 
                "paragraph_text": "Reference text", 
                "is_supporting": true
            },
            ...
        ]
    }
    """
    result = []
    for i, items in enumerate(data, start=1):
        answer_str = items['answer']
        contexts = []
        for context_id, reference in enumerate(items['references']):
            citation_str = f"[{context_id + 1}]"
            context_dict = {
                "idx": context_id,
                "title": None,
                "paragraph_text": reference,
                "is_supporting": True if citation_str in items['answer'] else False,
            }
            contexts.append(context_dict)
            answer_str = answer_str.replace(citation_str, "")

        answer_dict = {
            "number": "",
            "date": {"day": "", "month": "", "year": ""},
            "spans": [answer_str]
        }
        
        transformed = {
            "question_id": i,
            "question_text": items['question'],
            "answer_objects": [answer_dict],
            "contexts": contexts
        }
        result.append(transformed)
    return result


if __name__ == "__main__":

    input_directory = os.path.join(RAW_DATA, "webglmqa")
    
    if len(os.listdir(input_directory)) == 1:
        json_file = os.listdir(input_directory)[0]
        d_path = os.path.join(input_directory, json_file)
        f_name, _ = os.path.splitext(os.path.basename(d_path))
        if json_file.endswith('.json'):
            input_data = read_json(d_path)
        elif json_file.endswith('.jsonl'):
            input_data = read_jsonl(d_path)
    
        input_data = sample_jsonl(input_data)
        output_directory = os.path.join(PROCESSED_DATA, "webglmqa")
        os.makedirs(output_directory, exist_ok=True)
        
        output_path = os.path.join(output_directory, "test" + '.jsonl')
        output_data = transform_jsonl_data(input_data)
        
        write_jsonl(output_data, output_path)
    
    else:
        for json_file in os.listdir(input_directory):
            d_path = os.path.join(input_directory, json_file)
            f_name, _ = os.path.splitext(os.path.basename(d_path))
            if json_file.endswith('.json'):
                input_data = read_json(d_path)
            elif json_file.endswith('.jsonl'):
                input_data = read_jsonl(d_path)
        
            if f_name == "test":
                input_data = sample_jsonl(input_data)
            output_directory = os.path.join(PROCESSED_DATA, "webglmqa")
            os.makedirs(output_directory, exist_ok=True)
            
            output_path = os.path.join(output_directory, f_name + '.jsonl')
            output_data = transform_jsonl_data(input_data)
            
            write_jsonl(output_data, output_path)