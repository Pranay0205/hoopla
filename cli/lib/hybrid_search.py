import os
from pydoc import doc
import re

from lib.utils.constants import DEFAULT_SEARCH_LIMIT
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

        combined = combine_search_results(
            bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        print(f"Total fused results: {len(fused)}")
        return fused[:limit]


def normalize_scores(scores):
    """Normalize scores to 0-1 range using min-max normalization."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def normalize_search_results(results):
    """Add normalized_score field to each result."""
    scores = [result["score"] for result in results]
    normalized = normalize_scores(scores)

    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(bm25_score, semantic_score, alpha):
    """Calculate weighted hybrid score."""
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(bm25_results, semantic_results, alpha):
    """Combine BM25 and semantic search results using weighted scoring."""
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    # Process BM25 results - take MAX normalized score for duplicates
    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }

        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    # Process semantic results - take MAX normalized score for duplicates
    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }

        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    # Calculate hybrid scores
    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(
            data["bm25_score"], data["semantic_score"], alpha)
        result = {
            "id": doc_id,
            "title": data["title"],
            "document": data["document"],
            "score": score_value,
            "bm25_score": data["bm25_score"],
            "semantic_score": data["semantic_score"],
        }
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def rrf_score(rank, k):
    """Calculate reciprocal rank fusion score."""
    return 1 / (k + rank)


def reciprocal_rank_fusion(bm25_results, semantic_results, k):
    """Combine BM25 and semantic search results using reciprocal rank fusion."""
    rrf_scores = {}

    # Process BM25 results - only set rank once per document
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    # Process semantic results - only set rank once per document
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
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
            "score": data["rrf_score"],
            "rrf_score": data["rrf_score"],
            "bm25_rank": data["bm25_rank"],
            "semantic_rank": data["semantic_rank"],
        }
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)


def get_hybrid_search():
    documents = load_movies()
    return HybridSearch(documents)
