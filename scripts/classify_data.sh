#!/bin/bash

for dataset_name in nq trivia squad 2wikimultihopqa hotpotqa musique;
do
    # for dataset_type in "train" "dev" "test_subsampled" "dev_500_subsampled";
    for dataset_type in "train" "dev";
    do
        python classifier/run.py --dataset_name $dataset_name --dataset_type $dataset_type
    done
done

echo "Classification done"

for dataset in nq trivia squad 2wikimultihopqa hotpotqa musique
do
    for split in train dev
    do
        for type in DEBATE EVIDENCE-BASED INSTRUCTION REASON EXPERIENCE COMPARISON 
        do
            echo "\"$dataset/nf_$split.jsonl - $type\": $(cat $dataset/nf_$split.jsonl | grep $type | wc -l),"
        done
    done
done