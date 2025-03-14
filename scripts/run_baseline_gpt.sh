#!/bin/sh

export $(cat .env | xargs)
huggingface-cli login --token $HUGGINGFACE_TOKEN
export HF_HOME=$HUGGINGFACE_CACHE_DIR

# python retriever/build_index_wiki.py

for model_path in "gpt-4o-mini-2024-07-18"; do
    for method in "LLM" "RAG"; do
        for dataset_name in 2wikimultihopqa hotpotqa musique nq squad trivia; do
            python experiments/run_baseline.py --model_path $model_path --dataset_name $dataset_name --method $method
        done
    done
done