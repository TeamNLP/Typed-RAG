#!/bin/sh

export $(cat .env | xargs)
huggingface-cli login --token $HUGGINGFACE_TOKEN
export HF_HOME=$HUGGINGFACE_CACHE_DIR

# python retriever/build_index_wiki.py

for generator_model_path in "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2"; do
    for dataset_name in 2wikimultihopqa hotpotqa musique nq squad trivia; do
        CUDA_VISIBLE_DEVICES=0 python experiments/run_Typed-RAG.py \
            --generator_model_path $generator_model_path --dataset_name $dataset_name \
            --debug True --retriever_reindexing False --retriever_mode "existing"
    done
done