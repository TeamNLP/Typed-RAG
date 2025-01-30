#!/bin/bash

export $(cat .env | xargs)
huggingface-cli login --token $HUGGINGFACE_TOKEN
export HF_HOME=$HUGGINGFACE_CACHE_DIR

for scorer_model_name in "mistralai/Mistral-7B-Instruct-v0.2"; do
    # for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins" "gpt-4o-mini"; do
    for model_alias_to_evaluate in "gpt-4o-mini"; do
        python evaluation/LINKAGE.py \
            --model_alias_to_evaluate $model_alias_to_evaluate \
            --scorer_model_name $scorer_model_name
    done
done

