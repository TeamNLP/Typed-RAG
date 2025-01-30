#!/bin/bash

# construct reference list
for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    for model_path in mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3.1-8B-Instruct 
    do
        python data/reference_list_construction/run.py \
            --model_path $model_path \
            --dataset_name $dataset
    done
    
    for model_path in gpt-3.5-turbo-16k
    do
        python data/reference_list_construction/run.py \
            --model_path $model_path \
            --dataset_name $dataset
    done
done

cp -r data/processed_data/antique data/reference_list_construction/
cp -r data/processed_data/trecdlnf data/reference_list_construction/

# extract references
for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    python data/reference_list_construction/extract_references.py \
        --dataset_name $dataset
done