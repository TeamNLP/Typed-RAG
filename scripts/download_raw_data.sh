#!/bin/bash

echo "Downloading raw data for Typed-RAG"

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

mkdir -p .temp/
mkdir -p data/raw_data

echo "\n\nDownloading raw hotpotqa data\n"
mkdir -p data/raw_data/hotpotqa
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/raw_data/hotpotqa/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/raw_data/hotpotqa/hotpot_dev_distractor_v1.json

echo "\n\nDownloading raw 2wikimultihopqa data\n"
mkdir -p data/raw_data/2wikimultihopqa
wget https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0 -O .temp/2wikimultihopqa.zip
unzip -jo .temp/2wikimultihopqa.zip -d data/raw_data/2wikimultihopqa -x "*.DS_Store"
rm data_ids.zip*

echo "\n\nDownloading raw musique data\n"
mkdir -p data/raw_data/musique
# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O .temp/musique_v1.0.zip
unzip -jo .temp/musique_v1.0.zip -d data/raw_data/musique -x "*.DS_Store"

# echo "\n\nDownloading raw iirc data\n"
# mkdir -p data/raw_data/iirc
# wget https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz -O .temp/iirc_train_dev.tgz
# tar -xzvf .temp/iirc_train_dev.tgz -C .temp/
# mv .temp/iirc_train_dev/train.json data/raw_data/iirc/train.json
# mv .temp/iirc_train_dev/dev.json data/raw_data/iirc/dev.json

# echo "\n\nDownloading iirc wikipedia corpus (this will take 2-3 mins)\n"
# wget https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz -O .temp/context_articles.tar.gz
# tar -xzvf .temp/context_articles.tar.gz -C data/raw_data/iirc

# echo "\n\nDownloading hotpotqa wikipedia corpus (this will take ~5 mins)\n"
# wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O .temp/wikpedia-paragraphs.tar.bz2
# tar -xvf .temp/wikpedia-paragraphs.tar.bz2 -C data/raw_data/hotpotqa
# mv data/raw_data/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts data/raw_data/hotpotqa/wikpedia-paragraphs

rm -rf .temp/

echo "\n\nDownloading Natural Question\n"
mkdir -p data/raw_data/nq
cd data/raw_data/nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz

echo "\n\nDownloading TriviaQA\n"
cd ..
mkdir -p trivia
cd trivia
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
gzip -d biencoder-trivia-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
gzip -d biencoder-trivia-train.json.gz

echo "\n\nDownloading SQuAD\n"
cd ..
mkdir -p squad
cd squad
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
gzip -d biencoder-squad1-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
gzip -d biencoder-squad1-train.json.gz

echo "\n\nDownloading Wiki passages. For the singe-hop datasets, we use the Wikipedia as the document corpus.\n"
cd ..
mkdir -p wiki
cd wiki
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz

echo "\n\nDownloading ANTIQUE\n"
cd ..
mkdir -p antique
cd antique
wget https://raw.githubusercontent.com/babyyang525/LINKAGE-Listwise-NFQA-Evaluation/refs/heads/main/dataset/ANTIQUE.json

echo "\n\nDownloading TREC-DL-NF\n"
cd ..
mkdir -p trecdlnf
cd trecdlnf
wget https://raw.githubusercontent.com/babyyang525/LINKAGE-Listwise-NFQA-Evaluation/refs/heads/main/dataset/TREC-DL-NF.json

echo "\n\nDownloading WebGLM-QA\n"
cd ..
mkdir -p webglmqa
cd webglmqa
wget -O test.jsonl https://huggingface.co/datasets/THUDM/webglm-qa/resolve/main/data/test.jsonl?download=true
wget -O dev.jsonl https://huggingface.co/datasets/THUDM/webglm-qa/resolve/main/data/validation.jsonl?download=true
wget -O train.jsonl https://huggingface.co/datasets/THUDM/webglm-qa/resolve/main/data/train.jsonl?download=true

# The resulting data/raw_data/ directory should look like:
# .
# ├── 2wikimultihopqa
# │   ├── dev.json
# │   ├── id_aliases.json
# │   ├── test.json
# │   └── train.json
# ├── antique
# │   └── ANTIQUE.json
# ├── hotpotqa
# │   ├── hotpot_dev_distractor_v1.json
# │   └── hotpot_train_v1.1.json
# ├── musique
# │   ├── dev_test_singlehop_questions_v1.0.json
# │   ├── musique_ans_v1.0_dev.jsonl
# │   ├── musique_ans_v1.0_test.jsonl
# │   ├── musique_ans_v1.0_train.jsonl
# │   ├── musique_full_v1.0_dev.jsonl
# │   ├── musique_full_v1.0_test.jsonl
# │   └── musique_full_v1.0_train.jsonl
# ├── nq
# │   ├── biencoder-nq-dev.json
# │   └── biencoder-nq-train.json
# ├── squad
# │   ├── biencoder-squad1-dev.json
# │   └── biencoder-squad1-train.json
# ├── trecdlnf
# │   └── TREC-DL-NF.json
# ├── trivia
# │   ├── biencoder-trivia-dev.json
# │   └── biencoder-trivia-train.json
# ├── webglmqa
# │   ├── dev.jsonl
# │   ├── test.jsonl
# │   └── train.jsonl
# └── wiki
#     └── psgs_w100.tsv
