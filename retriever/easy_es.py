from easy_elasticsearch import ElasticSearchBM25

pool = {
    "id1": "What is Python? Is it a programming language",
    "id2": "Which Python version is the best?",
    "id3": "Using easy-elasticsearch in Python is really convenient!",
}
bm25 = ElasticSearchBM25(
    pool, port_http="9222", port_tcp="9333"
)  # By default, when `host=None` and `mode="docker"`, a ES docker container will be started at localhost.

query = "What is Python?"
rank = bm25.query(query, topk=10)  # topk should be <= 10000
scores = bm25.score(query, document_ids=["id2", "id3"])

print(query, rank, scores)
print(rank)
bm25.delete_index()  # delete the one-trial index named 'one_trial'
bm25.delete_container()  # remove the docker container'
