import csv
import json
import os
from tqdm import tqdm

data = {}

raw_glob_filepath = os.path.join("data", "raw_data", "wiki", "psgs_w100.tsv")

with open(raw_glob_filepath, "r", encoding="utf-8") as tsv_file:
    tsv_reader = csv.DictReader(tsv_file, delimiter="\t")
    for row in tqdm(tsv_reader):
        data[row["id"]] = row["text"]

wiki_corpus_glob_filepath = os.path.join("retriever", "wiki_corpus.json")

with open(wiki_corpus_glob_filepath, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)
