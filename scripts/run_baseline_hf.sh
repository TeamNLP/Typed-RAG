#!/bin/sh

export $(cat .env | xargs)
huggingface-cli login --token $HUGGINGFACE_TOKEN
export HF_HOME=$HUGGINGFACE_CACHE_DIR

# python retriever/build_index_wiki.py

for model_path in "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2"; do
    for method in "LLM" "RAG"; do
        for dataset_name in 2wikimultihopqa hotpotqa musique nq squad trivia; do
            python experiments/run_baseline.py --model_path $model_path --dataset_name $dataset_name --method $method
        done
    done
done