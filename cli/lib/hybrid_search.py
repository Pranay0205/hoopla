import os
from pydoc import doc
import re

from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.hybrid_search_utils import hybrid_score, normalize_scores
from lib.utils.math_utils import rrf_score
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
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500)

        # Normalize scores
        bm25_scores = [x["score"] for x in bm25_results]
        normalized_bm25_scores = normalize_scores(bm25_scores)

        semantic_scores = [x["score"] for x in semantic_results]
        normalized_semantic_scores = normalize_scores(semantic_scores)

        # Build combined scores dictionary
        combined_scores = {}

        # Process BM25 results - take MAX normalized score for duplicates
        for i, res in enumerate(bm25_results):
            doc_id = res["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "title": res["title"],
                    "document": res["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            # Take maximum score if document appears multiple times
            if normalized_bm25_scores[i] > combined_scores[doc_id]["bm25_score"]:
                combined_scores[doc_id]["bm25_score"] = normalized_bm25_scores[i]

        # Process semantic results - take MAX normalized score for duplicates
        for i, res in enumerate(semantic_results):
            doc_id = res["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "title": res["title"],
                    "document": res["document"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }
            # Take maximum score if document appears multiple times
            if normalized_semantic_scores[i] > combined_scores[doc_id]["semantic_score"]:
                combined_scores[doc_id]["semantic_score"] = normalized_semantic_scores[i]

        # Calculate hybrid scores
        hybrid_results = []
        for doc_id, data in combined_scores.items():
            score_value = hybrid_score(
                data["bm25_score"], data["semantic_score"], alpha)
            result = {
                "id": doc_id,
                "title": data["title"],
                "document": data["document"],
                "score": score_value,  # Final hybrid score
                "bm25_score": data["bm25_score"],
                "semantic_score": data["semantic_score"],
            }
            hybrid_results.append(result)

        # Sort by hybrid score and return top results
        sorted_results = sorted(
            hybrid_results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500)

        rrf_scores = {}

        # Process BM25 results - only set rank once per document
        for rank, res in enumerate(bm25_results, start=1):
            doc_id = res["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "title": res["title"],
                    "document": res["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
            # Only set rank if not already set (first occurrence)
            if rrf_scores[doc_id]["bm25_rank"] is None:
                rrf_scores[doc_id]["bm25_rank"] = rank
                rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

        # Process semantic results - only set rank once per document
        for rank, res in enumerate(semantic_results, start=1):
            doc_id = res["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "title": res["title"],
                    "document": res["document"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }
            # Only set rank if not already set (first occurrence)
            if rrf_scores[doc_id]["semantic_rank"] is None:
                rrf_scores[doc_id]["semantic_rank"] = rank
                rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

        # Build final results
        rrf_results = []
        for doc_id, data in rrf_scores.items():
            result = {
                "id": doc_id,
                "title": data["title"],
                "document": data["document"],
                "score": data["rrf_score"],  # Final RRF score
                "rrf_score": data["rrf_score"],
                "bm25_rank": data["bm25_rank"],
                "semantic_rank": data["semantic_rank"],
            }
            rrf_results.append(result)

        # Sort by RRF score and return top results
        sorted_results = sorted(
            rrf_results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]


def get_hybrid_search() -> HybridSearch:
    documents = load_movies()
    return HybridSearch(documents)
