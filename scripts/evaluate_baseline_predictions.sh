#!/bin/bash

model_alias="gpt-4o-mini"
for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    python evaluation/evaluate.py \
        --model_alias $model_alias \
        --dataset_name $dataset
done