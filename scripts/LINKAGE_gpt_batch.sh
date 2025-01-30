#!/bin/bash

for scorer_model_name in "gpt-4o-mini-2024-07-18"; do
    # Create OpenAI Batch API 
    # for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins" "gpt-4o-mini"; do
    #     python evaluation/LINKAGE.py \
    #         --use_openai_batch_api \
    #         --model_alias_to_evaluate $model_alias_to_evaluate \
    #         --scorer_model_name $scorer_model_name \
    #         --openai_batch_api_mode "create"
    # done

    # List OpenAI Batch API
    # for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins" "gpt-4o-mini"; do
    for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins"; do
        python evaluation/LINKAGE.py \
            --use_openai_batch_api \
            --model_alias_to_evaluate $model_alias_to_evaluate \
            --scorer_model_name $scorer_model_name \
            --openai_batch_api_mode "list"
    done

    # Retrieve OpenAI Batch API
    # for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins" "gpt-4o-mini"; do
    for model_alias_to_evaluate in "mistral-7b-ins" "llama-3.2-3b-ins"; do
        python evaluation/LINKAGE.py \
            --use_openai_batch_api \
            --model_alias_to_evaluate $model_alias_to_evaluate \
            --scorer_model_name $scorer_model_name \
            --openai_batch_api_mode "retrieve"
    done
done