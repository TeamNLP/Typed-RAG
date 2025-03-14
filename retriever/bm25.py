import os
import json
from typing import List, Dict, Any

from easy_elasticsearch import ElasticSearchBM25


def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf8", errors="ignore") as file:
        instance = json.load(file)
    return instance


class BM25Retriever:
    def __init__(self, corpus_name, top_n=5, reindexing=False, port_http="9200", port_tcp="9300", mode="existing"):
        self.corpus_name = corpus_name
        self.top_n = top_n
        self.mode = mode

        assert self.top_n <= 10000, "top_n should be <= 10000"

        if self.corpus_name == "wiki":
            print("Start reading the corpus data...")
            print("It will take few minutes...")
            # chunks_glob_filepath = os.path.join(
            #     os.path.dirname(os.path.realpath(__file__)), "wiki_corpus.json"
            # )
            chunks_glob_filepath = "/data3/donggeonlee/Typed-RAG/retriever/wiki_corpus.json"

            corpus = read_json(chunks_glob_filepath)
            print("End reading the corpus data...")
        else:
            raise NotImplementedError

        if self.mode == "docker":
            self.bm25 = ElasticSearchBM25(
                corpus,
                index_name="wiki",
                reindexing=reindexing,
                port_http=port_http,
                port_tcp=port_tcp,
                service_type="docker",
            )
        elif self.mode == "executable":
            self.bm25 = ElasticSearchBM25(
                corpus,
                index_name="wiki",
                reindexing=reindexing,
                port_http=port_http,
                port_tcp=port_tcp,
                service_type="executable",
            )
        else:
            # Or use an existing ES service:
            assert self.mode == "existing"
            self.bm25 = ElasticSearchBM25(
                corpus,
                index_name="wiki",
                reindexing=reindexing,
                host="http://localhost",
                port_http=port_http,
                port_tcp=port_tcp,
                service_type="existing",
            )


    def __del__(self):
        pass
        # self.bm25.delete_index() # delete the one-trial index named 'one_trial'
        # if self.mode == "docker":
        #     self.bm25.delete_container()
        # elif self.mode == "executable":
        #     self.bm25.delete_excutable()

    def get_bm25_meta_result(
        self,
        query: str,
    ) -> Dict[str, str]:
        """
        Retrieve the top-n documents for the given query and return the meta information

        Args:
            query: query string

        Returns:
            meta_result: dictionary of top-n documents (key: document id, value: document text)
        """
        meta_result = self.bm25.query(query, topk=self.top_n)
        return meta_result

    def retrieve(
        self,
        query: str,
        return_type: str = "text",
    ) -> Dict[str, str]:
        """
        Retrieve the top-n documents for the given query and return the result

        Args:
            query: query string

        Returns:
            references_text or references_list: Union[str, List[str]] - references text or references list
        """
        passages_dict = self.get_bm25_meta_result(query)

        if return_type.lower() in ["text", "str", "string"]:
            references_text = ""
            for passage in passages_dict:
                assert isinstance(passage, str)
                references_text += passage.strip()
                references_text += "\n"
            references_text = references_text.strip()
            return references_text

        elif return_type.lower() in ["list"]:
            references_list = [passage.strip() for passage in passages_dict]
            
            assert len(references_list) == self.top_n, print(f"The number of retrieved documents is less than top_n.\n\
                The number of retrieved documents: {len(references_list)}\n\
                The top_n: {self.top_n}\n\
                Query: {query}\n\
                Please check the corpus data or the BM25 retriever.\n\
                The references_list {references_list}\n\
                The passages_dict: \n{passages_dict}\
            ")

            return references_list


# if __name__ == "__main__":
#     retriever = BM25Retriever(corpus_name='wiki', top_n=5, reindexing=False)
#     query = "What is the capital of South Korea?"

#     rank = retriever.retrieve(query)
#     print(rank)
