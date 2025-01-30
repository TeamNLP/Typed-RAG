import os

from typing import List
from config import PROCESSED_DATA, RAW_DATA
from lib import read_json, read_jsonl, write_jsonl


def normalize_answer_label(answer_label: List[int]):
    """
    `4/3/2/1` -> `3/2/1/0`
    Normalize answer label to 0-based index based on the original LINKAGE paper.
    """
    return [label - 1 for label in answer_label]

def transform_data(data):
    """
    output jsonl format:
    {
        "question_id": 1,
        "question_text": "how can we get concentration on something?",
        "answer_list": ["answer1", "answer2", "answer3"],
        "answer_label": [4, 4, 4]
    }
    """
    result = []
    for i, (question_text, answers) in enumerate(data.items(), start=1):
        transformed = {
            "question_id": i,
            "question_text": question_text,
            "answer_list": [answer["passage"] for answer in answers],
            "answer_label": normalize_answer_label([answer["label"] for answer in answers])
        }
        result.append(transformed)

    return result


if __name__ == "__main__":

    input_directory = os.path.join(RAW_DATA, "antique")
    
    if len(os.listdir(input_directory)) == 1:
        json_file = os.listdir(input_directory)[0]
        d_path = os.path.join(input_directory, json_file)
        f_name, _ = os.path.splitext(os.path.basename(d_path))
        if json_file.endswith('.json'):
            input_data = read_json(d_path)
        elif json_file.endswith('.jsonl'):
            input_data = read_jsonl(d_path)
        
        output_directory = os.path.join(PROCESSED_DATA, "antique")
        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(output_directory, "test" + '.jsonl')
        output_data = transform_data(input_data)

        write_jsonl(output_data, output_path)

    else:
        for json_file in os.listdir(input_directory):
            d_path = os.path.join(input_directory, json_file)
            f_name, _ = os.path.splitext(os.path.basename(d_path))
            if json_file.endswith('.json'):
                input_data = read_json(d_path)
            elif json_file.endswith('.jsonl'):
                input_data = read_jsonl(d_path)
        
            output_directory = os.path.join(PROCESSED_DATA, "antique")
            os.makedirs(output_directory, exist_ok=True)
            
            output_path = os.path.join(output_directory, f_name + '.jsonl')
            output_data = transform_data(input_data)
            
            write_jsonl(output_data, output_path)