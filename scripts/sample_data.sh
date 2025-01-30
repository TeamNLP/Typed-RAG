#!/bin/bash

cd data

# Subsample and Filter the processed nf_datasets
for dataset_name in nq trivia squad hotpotqa 2wikimultihopqa musique
do
    python processing/subsample_nf_dataset_and_remap_paras.py $dataset_name test 300
    python processing/filter_nf_dataset_and_remap_paras.py $dataset_name test 300
done

