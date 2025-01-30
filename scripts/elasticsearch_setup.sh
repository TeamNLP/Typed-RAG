#!/bin/bash

#### Downloading elasticsearch ####
ES=./retriever/elasticsearch-7.9.1/bin/elasticsearch
if test -f "$ES"; then
    echo "$ES exists. Using the existent one"
else 
    echo "$ES does not exist. Downloading a new one"
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.1-linux-x86_64.tar.gz -P retriever
    tar -xf retriever/elasticsearch-7.9.1-linux-x86_64.tar.gz -C retriever
fi

#### Starting the ES service ####
nohup ./retriever/elasticsearch-7.9.1/bin/elasticsearch > elasticsearch.log &

#### Building the index and Running the example #### 
python retriever/build_index_wiki.py