import os
from json.decoder import JSONDecodeError

from config import RAW_DATA, PROCESSED_DATA
from lib import read_json, read_jsonl, write_jsonl


def main():

    set_names = ["train", "dev"]

    input_directory = os.path.join(RAW_DATA, "hotpotqa")
    output_directory = os.path.join(PROCESSED_DATA, "hotpotqa")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        if set_name == "train":
            input_filepath = os.path.join(input_directory, f"hotpot_train_v1.1.json")

        elif set_name == "dev":
            input_filepath = os.path.join(input_directory, f"hotpot_dev_distractor_v1.json")

        raw_instances = read_json(input_filepath)
        for raw_instance in raw_instances:
            
            question_id = raw_instance["_id"]
            question_text = raw_instance["question"]
            raw_contexts = raw_instance["context"]

            supporting_titles = list(set([e[0] for e in raw_instance["supporting_facts"]]))

            processed_contexts = []
            for index, raw_context in enumerate(raw_contexts):
                title = raw_context[0]
                paragraph_text = " ".join(raw_context[1]).strip()
                is_supporting = title in supporting_titles
                processed_contexts.append(
                    {
                        "idx": index,
                        "title": title.strip(),
                        "paragraph_text": paragraph_text,
                        "is_supporting": is_supporting,
                    }
                )

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            answers_objects = [answers_object]

            processed_instance = {
                "question_id": question_id,
                "question_text": question_text,
                "answers_objects": answers_objects,
                "contexts": processed_contexts,
            }

            processed_instances.append(processed_instance)

        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")
        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
