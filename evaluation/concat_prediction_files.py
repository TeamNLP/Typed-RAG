from lib import read_jsonl, write_jsonl


def get_reference_instances(dataset_name):
    dataset_name_to_path = {
        "2wikimultihopqa": "data/processed_data/2wikimultihopqa/nf_test_79_filtered.jsonl",
        "hotpotqa": "data/processed_data/hotpotqa/nf_test_62_filtered.jsonl",
        "musique": "data/processed_data/musique/nf_test_82_filtered.jsonl",
        "nq": "data/processed_data/nq/nf_test_126_filtered.jsonl",
        "squad": "data/processed_data/squad/nf_test_300_filtered.jsonl",
        "trivia": "data/processed_data/trivia/nf_test_296_filtered.jsonl",
    }
    reference_data_path = dataset_name_to_path[dataset_name]
    return read_jsonl(reference_data_path)

instance_key_order = [
    "dataset",
    "NFQ_category",
    "question_id",
    "question_text",
    "output",
    "answer_list",
    "answer_label",
    "reference_answers",
    "LINKAGE_rank",
    "LINKAGE_normalized_rank"
]

evaluation_results_path_list = ["evaluation/LINKAGE_results/llama-3.2-3b-ins", "evaluation/LINKAGE_results/mistral-7b-ins"]
dataset_name_list = ["nq", "trivia", "squad", "2wikimultihopqa", "hotpotqa", "musique"]
prediction_file_prefix_list = ["LLM_prediction_", "RAG_prediction_", "Typed-RAG_prediction_"]
prediction_file_suffix_list = ["_annotated_odqa_nf_test_LINKAGE_results_by_gpt-4o-mini", "_annotated_odqa_nf_test_LINKAGE_results_by_mistral-7b-ins"]

for evaluation_results_path in evaluation_results_path_list:
    for prefix in prediction_file_prefix_list:
        for suffix in prediction_file_suffix_list:
            output_instances = []
            for dataset_name in dataset_name_list:
                prediction_file_name = f"{prefix}{dataset_name}{suffix}"
                prediction_file_path = f"{evaluation_results_path}/{prediction_file_name}.jsonl"
                input_instances = read_jsonl(prediction_file_path)

                reference_instances = get_reference_instances(dataset_name)
                for input_instance in input_instances:
                    question_id = input_instance["question_id"]
                    NFQ_category = None
                    for reference_instance in reference_instances:
                        if reference_instance["question_id"] == question_id:
                            NFQ_category = reference_instance["category_prediction"]
                            break
                    assert NFQ_category is not None, f"NFQ_category is None. question_id: {question_id}"

                    input_instance["dataset"] = dataset_name
                    input_instance["NFQ_category"] = NFQ_category

                    output_instances.append(dict(sorted(input_instance.items(), key=lambda k: instance_key_order.index(k[0]))))

            output_file_name = f"{prefix}Wiki-NFQA{suffix}.jsonl"
            output_file_path = f"{evaluation_results_path}/{output_file_name}"
            write_jsonl(output_instances, output_file_path)