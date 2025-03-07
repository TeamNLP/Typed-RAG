#!/bin/bash

# construct reference list
for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    for model_path in mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3.2-3B-Instruct
    do
        python data/reference_list_construction/run.py \
            --model_path $model_path \
            --dataset_name $dataset
    done
    
    for model_path in gpt-4o-mini-2024-07-18
    do
        python data/reference_list_construction/run.py \
            --model_path $model_path \
            --dataset_name $dataset
    done
done

# extract references
for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    python data/reference_list_construction/extract_references.py \
        --dataset_name $dataset
done