#!/bin/sh

export $(cat .env | xargs)
huggingface-cli login --token $HUGGINGFACE_TOKEN
export HF_HOME=$HUGGINGFACE_CACHE_DIR

# python retriever/build_index_wiki.py

for generator_model_path in "gpt-4o-mini-2024-07-18"; do
    for dataset_name in 2wikimultihopqa hotpotqa musique nq squad trivia; do
        python experiments/run_Typed-RAG.py \
            --use_gpt True \
            --generator_model_path $generator_model_path --dataset_name $dataset_name \
            --debug True --retriever_reindexing False --retriever_mode "existing"
    done
done