import os
import re

from lib.utils.hybrid_search_utils import hybrid_score, normalize_scores
from lib.utils.search_utils import load_movies

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)

        semantic_results = self.semantic_search.search(query, limit * 500)

        bm25_scores = [x["score"] for x in bm25_results]

        normalized_bm25_scores = normalize_scores(bm25_scores)

        semantic_scores = [x["score"] for x in semantic_results]

        normalized_semantic_scores = normalize_scores(semantic_scores)

        documents = {}

        for i, res in enumerate(bm25_results):
            documents[res["id"]] = {**res}
            documents[res["id"]]["keyword_score"] = normalized_bm25_scores[i]
            documents[res["id"]]["semantic_score"] = 0.0

        for i, res in enumerate(semantic_results):
            if res["id"] in documents:
                documents[res["id"]
                          ]["semantic_score"] = normalized_semantic_scores[i]
            else:

                documents[res["id"]] = {**res}
                documents[res["id"]]["keyword_score"] = 0.0
                documents[res["id"]
                          ]["semantic_score"] = normalized_semantic_scores[i]

        for doc_id in documents:
            kw_score = documents[doc_id]["keyword_score"]
            sem_score = documents[doc_id]["semantic_score"]
            documents[doc_id]["hybrid_score"] = hybrid_score(
                kw_score, sem_score, alpha)

        sorted_results = sorted(
            documents.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return sorted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def get_hybrid_search() -> HybridSearch:
    documents = load_movies()

    return HybridSearch(documents)
