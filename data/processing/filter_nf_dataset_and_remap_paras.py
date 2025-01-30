import argparse
import random
import os

from collections import defaultdict
from typing import List, Dict

from tqdm import tqdm
from lib import read_jsonl, write_jsonl


SAMPLE_SIZE = 300
NUM_NF_CLASSES = 6
NUM_REQUIRED = SAMPLE_SIZE // NUM_NF_CLASSES

random.seed(13370)  # Don't change this.


def sample_dataset(
        data: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    
    categorized_data = defaultdict(list)
    for item in data:
        category = item['category_prediction']
        categorized_data[category].append(item)

    sampled_data = []
    remaining_data = defaultdict(list)
    for category, items in tqdm(categorized_data.items()):
        if len(items) >= NUM_REQUIRED:
            sampled_data.extend(random.sample(items, NUM_REQUIRED))
            remaining_data[category].extend([item for item in items if item not in sampled_data])
        else:
            sampled_data.extend(items)

    return sampled_data, remaining_data


def complete_sampling(
        data: List[Dict[str, str]], 
        sample_size=300
) -> List[Dict[str, str]]:
    sampled_data, remaining_data = sample_dataset(data)
    total_samples = len(sampled_data)
    required_samples = sample_size - total_samples

    if required_samples > 0:
        all_remaining = [item for items in remaining_data.values() for item in items]
        additional_samples = random.sample(all_remaining, min(required_samples, len(all_remaining)))
        sampled_data.extend(additional_samples)

    return sampled_data


def main():
    parser = argparse.ArgumentParser(description="Save and sample data")
    parser.add_argument(
        "dataset_name", type=str, help="dataset name.", choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad')
    )
    parser.add_argument("set_name", type=str, help="set name.", choices=("dev", "test", "dev_diff_size"))
    parser.add_argument("sample_size", type=int, help="sample_size")
    args = parser.parse_args()

    avoid_question_ids_file_path = None
    sample_size = SAMPLE_SIZE
    if args.set_name == "test":
        dev_file_path = os.path.join("processed_data", args.dataset_name, "nf_dev_subsampled.jsonl")
        avoid_question_ids_file_path = dev_file_path if os.path.exists(dev_file_path) else None
        sample_size = SAMPLE_SIZE
    
    if args.set_name == "dev_diff_size":
        avoid_question_ids_file_path = os.path.join("processed_data", args.dataset_name, "nf_test_subsampled.jsonl")
        sample_size = args.sample_size

    input_file_path = os.path.join("processed_data", args.dataset_name, "nf_dev.jsonl")
    instances = read_jsonl(input_file_path)

    if avoid_question_ids_file_path:
        avoid_ids = set([avoid_instance["question_id"] for avoid_instance in read_jsonl(avoid_question_ids_file_path)])
        instances = [instance for instance in instances if instance["question_id"] not in avoid_ids]

    filtered_instances = []
    for instance in instances:
        if "how" in instance["question_text"].lower():
            for how_pattern in ["how to", "how can i", "how come", "happened?", "work?"," describe ", " to ", " from ", " against ", "how do you like"]:
                if how_pattern in instance["question_text"].lower():
                    filtered_instances.append(instance)
                    break
        elif "what" in instance["question_text"].lower():
            for what_pattern in [" preocess", " way", " reason", "what causes", "what is", "what are", "what was", "what were", " property", " properties", " meaning of ", "over", " think about "]:
                if what_pattern in instance["question_text"].lower():
                    filtered_instances.append(instance)
                    break
        else:                
            for pattern in ["why", " recommend", "will you ", "would you ", "should ", " exist", " successful?", " really "]:
                if pattern in instance["question_text"].lower():
                    filtered_instances.append(instance)
                    break
    instances = filtered_instances
    
    # instances = random.sample(instances, sample_size)
    instances = complete_sampling(instances, sample_size)

    if args.set_name == "dev_diff_size":
        output_file_path = os.path.join("processed_data", args.dataset_name, f"nf_dev_{len(instances)}_filtered.jsonl")
        write_jsonl(instances, output_file_path)
    else:
        output_file_path = os.path.join("processed_data", args.dataset_name, f"nf_{args.set_name}_{len(instances)}_filtered.jsonl")
        write_jsonl(instances, output_file_path)


if __name__ == "__main__":
    main()
