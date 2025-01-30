import os
import csv

from lib import read_json


def extract_scores_by_methods(results_directory, methods, scorer_model):
    scores = {method: {} for method in methods}

    for filename in os.listdir(results_directory):
        if filename.endswith("_scores.json"):
            if "Wiki-NFQA" in filename:
                continue
            method, dataset = filename.split("_prediction_")
            dataset = dataset.replace("_scores.json", "")
            method = method.split("_")[0]
            dataset = dataset.split("_")[0]
            assert method in methods, f"Unknown method: {method}"

            result_instance = read_json(os.path.join(results_directory, filename))
            
            assert scorer_model in result_instance, f"Scorer model {scorer_model} not found in {filename}"
            metrics = result_instance[scorer_model]

            assert "MPR" in metrics, f"MPR not found in {filename}"
            assert "MRR" in metrics, f"MRR not found in {filename}"
            
            scores[method][dataset] = metrics # e.g., {"MPR": 0.65, "MRR": 0.71}

    return scores


def extract_scores_by_categories(results_directory, methods, categories, scorer_model):
    scores = {method: {} for method in methods}

    for filename in os.listdir(results_directory):
        if filename.endswith("_scores.json"):
            if "Wiki-NFQA" not in filename:
                continue
            method, category = filename.split("_prediction_")
            category = category.replace("_scores.json", "")
            method = method.split("_")[0]
            category = category.split("_")[-1]
            assert method in methods, f"Unknown method: {method}"

            result_instance = read_json(os.path.join(results_directory, filename))
            
            assert scorer_model in result_instance, f"Scorer model {scorer_model} not found in {filename}"
            metrics = result_instance[scorer_model]

            assert "MPR" in metrics, f"MPR not found in {filename}"
            assert "MRR" in metrics, f"MRR not found in {filename}"
            
            scores[method][category] = metrics # e.g., {"MPR": 0.65, "MRR": 0.71}

    return scores


def write_csv(scores, output_file, metric):
    assert output_file.endswith(".csv"), "Output file must be a CSV file"

    datasets = set()
    for method_scores in scores.values():
        datasets.update(method_scores.keys())

    datasets = sorted(datasets)

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Method"] + datasets
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for method, method_scores in scores.items():
            row = {"Method": method}
            for dataset in datasets:
                row[dataset] = method_scores.get(dataset, {}).get(metric, "")
            writer.writerow(row)

# main
if __name__ == "__main__":
    METHODS = ["LLM", "RAG", "Typed-RAG"]
    CATEGORIES = [
        "EVIDENCE-BASED",
        "REASON",
        "COMPARISON",
        "INSTRUCTION",
        "EXPERIENCE",
        "DEBATE"
    ]
    
    for scorer_model in ["mistral-7b-ins", "gpt-4o-mini"]:
        for target_model in ["llama-3.2-3b-ins", "mistral-7b-ins", "gpt-4o-mini"]:
            results_directory = f"evaluation/LINKAGE_results/{target_model}"

            scores_by_methods = extract_scores_by_methods(results_directory, METHODS, scorer_model)

            write_csv(scores_by_methods, f"evaluation/LINKAGE_results/{target_model}_MPR_by_{scorer_model}.csv", "MPR")
            write_csv(scores_by_methods, f"evaluation/LINKAGE_results/{target_model}_MRR_by_{scorer_model}.csv", "MRR")

            scores_by_categories = extract_scores_by_categories(results_directory, METHODS, CATEGORIES, scorer_model)

            write_csv(scores_by_categories, f"evaluation/LINKAGE_results/{target_model}_MPR_by_{scorer_model}_categories.csv", "MPR")
            write_csv(scores_by_categories, f"evaluation/LINKAGE_results/{target_model}_MRR_by_{scorer_model}_categories.csv", "MRR")

    