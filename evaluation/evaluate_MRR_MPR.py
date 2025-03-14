import os
from typing import Dict, Any, List

from lib import (
    read_json,
    read_jsonl,
    write_json,
)


def calculate_MRR(result_instance: List[Dict[str, Any]]) -> float:
    sum_rr = 0
    for instance in result_instance:
        rank = instance["LINKAGE_rank"]
        rr = 1 / rank
        sum_rr += rr

    length = len(result_instance)
    MRR = sum_rr / length
    return MRR


def calculate_MPR(result_instance: List[Dict[str, Any]]) -> float:
    """
    Calculate Mean Percentile Rank (MPR)
    
    MPR = (1 / N) * sum( (1 - (rank - 1) / total) * 100 )
    
    Args:
        result_instance (List[Dict[str, Any]]): 
            A list of dictionaries, each containing:
                - "LINKAGE_rank": The rank of the correct answer (1-based index).
                - "total_candidates": The total number of candidates.
    
    Returns:
        float: The Mean Percentile Rank (MPR).
    """
    sum_pr = 0
    for instance in result_instance:
        rank = instance["LINKAGE_rank"]
        total = len(instance["reference_answers"])
        percentile_rank = (1 - (rank - 1) / total) * 100
        sum_pr += percentile_rank

    length = len(result_instance)
    MPR = sum_pr / length
    return MPR


def main():
    METHODS = ["LLM", "RAG", "Typed-RAG"]
    CATEGORIES = [
        "EVIDENCE-BASED",
        "REASON",
        "COMPARISON",
        "INSTRUCTION",
        "EXPERIENCE",
        "DEBATE"
    ]
        
    for target_model in ["llama-3.2-3b-ins", "mistral-7b-ins", "gpt-4o-mini"]:
        results_directory = f"evaluation/LINKAGE_results/{target_model}"

        jsonl_file_list = os.listdir(results_directory)
        jsonl_file_list = [f for f in jsonl_file_list if f.endswith(".jsonl")]
        assert len(jsonl_file_list) > 0, f"No JSONL files found in {results_directory}"

        for jsonl_file_path in jsonl_file_list:
            method, temp_str = jsonl_file_path.split("_prediction_")
            assert method in METHODS, f"Unknown method: {method}"

            dataset, scorer_model = temp_str.split("_LINKAGE_results_by_")
            scorer_model = scorer_model.replace(".jsonl", "")
            assert scorer_model in ["mistral-7b-ins", "gpt-4o-mini", "gpt-4o"], f"Unknown scorer model: {scorer_model}"
            result_instance = read_jsonl(os.path.join(results_directory, jsonl_file_path))

            if dataset == "Wiki-NFQA_annotated_odqa_nf_test":
                # Separate by category (result_instance["NFQ_category"])
                for category in CATEGORIES:
                    category_result_instance = [instance for instance in result_instance if instance["NFQ_category"] == category]
                    try:
                        MRR_score = calculate_MRR(category_result_instance)
                        MPR_score = calculate_MPR(category_result_instance)
                    except ZeroDivisionError:
                        print(f"ZeroDivisionError: {target_model}/{jsonl_file_path}")
                        exit()

                    scores_json_file_name = f"{method}_prediction_{dataset}_{category}_scores.json"
                    scores_json_file_path = os.path.join(results_directory, scores_json_file_name)
                    
                    try:
                        assert os.path.exists(scores_json_file_path), f"No scores JSON file found: {scores_json_file_path}"
                    except AssertionError:
                        scores_json = {
                            "mistral-7b-ins": {"MRR": None, "MPR": None},
                            "gpt-4o-mini": {"MRR": None, "MPR": None},
                            "gpt-4o": {"MRR": None, "MPR": None},
                        }
                        write_json(scores_json, scores_json_file_path)
                    scores_json = read_json(scores_json_file_path)

                    scores_json[scorer_model]["MRR"] = MRR_score
                    scores_json[scorer_model]["MPR"] = MPR_score
                    write_json(scores_json, scores_json_file_path)
            else:
                try:
                    MRR_score = calculate_MRR(result_instance)
                    MPR_score = calculate_MPR(result_instance)
                except ZeroDivisionError:
                    print(f"ZeroDivisionError: {target_model}/{jsonl_file_path}")
                    exit()

                scores_json_file_name = f"{method}_prediction_{dataset}_scores.json"
                scores_json_file_path = os.path.join(results_directory, scores_json_file_name)
                
                assert os.path.exists(scores_json_file_path), f"No scores JSON file found: {scores_json_file_path}"
                scores_json = read_json(scores_json_file_path)

                scores_json[scorer_model]["MRR"] = MRR_score
                scores_json[scorer_model]["MPR"] = MPR_score
                write_json(scores_json, scores_json_file_path)


if __name__ == "__main__":
    main()