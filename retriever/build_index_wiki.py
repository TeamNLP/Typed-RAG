import argparse
import os

from easy_elasticsearch import ElasticSearchBM25
from lib import read_json

argparser = argparse.ArgumentParser()
argparser.add_argument("--host", type=str, default=None)
argparser.add_argument("--mode", type=str, default="docker")
argparser.add_argument("--port_http", type=int, default=9200)
argparser.add_argument("--port_tcp", type=int, default=9300)
args = argparser.parse_args()


print("Start reading the corpus data...")
print("It will take few minutes...")
wiki_corpus_glob_filepath = os.path.join("retriever", "wiki_corpus.json")
corpus = read_json(wiki_corpus_glob_filepath)
print("End reading the corpus data...")

bm25 = ElasticSearchBM25(
    corpus, index_name="wiki", reindexing=True, port_http=args.port_http, port_tcp=args.port_tcp, host=args.host, service_type=args.mode
)  # By default, when `host=None` and `mode="docker"`, a ES docker container will be started at localhost.

query = "What is Python?"
rank = bm25.query(query, topk=10)  # topk should be <= 10000

print("Query:", query)
# print("Rank:", rank)
