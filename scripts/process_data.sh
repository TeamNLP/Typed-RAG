#!/bin/bash

cd data

# Process raw data files in a single standard format
python processing/process_nq.py
python processing/process_trivia.py
python processing/process_squad.py
python processing/process_2wikimultihopqa.py
python processing/process_musique.py
python processing/process_hotpotqa.py

# The resulting experiments/processed_data/ directory should look like:
# .
# ├── 2wikimultihopqa
# │   ├── dev.jsonl
# │   └── train.jsonl
# ├── hotpotqa
# │   ├── dev.jsonl
# │   └── train.jsonl
# ├── musique
# │   ├── dev.jsonl
# │   └── train.jsonl
# ├── nq
# │   ├── dev.jsonl
# │   └── train.jsonl
# ├── squad
# │   ├── dev.jsonl
# │   └── train.jsonl
# └── trivia
#     ├── dev.jsonl
#     └── train.jsonl