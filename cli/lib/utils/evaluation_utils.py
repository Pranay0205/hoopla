

from lib.hybrid_search import HybridSearch
from lib.semantic_search import SemanticSearch
from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.search_utils import load_golden_dataset, load_movies


def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
                   ) -> float:
    relevant_count = 0

    for doc in retrieved_docs[:k]:
        if doc in relevant_docs:
            relevant_count += 1

    return relevant_count / k


def evaluate_f1_score(precision: float, recall: float) -> float:

    return 2 * (precision * recall) / (precision + recall)


def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    relevant_count = 0
    for doc in retrieved_docs[:k]:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)


def evaluate_command(limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    golden_dataset = load_golden_dataset()
    test_cases = golden_dataset["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query = {}

    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_documents = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_documents.append(title)

        precision = precision_at_k(retrieved_documents, relevant_docs, limit)
        recall = recall_at_k(retrieved_documents, relevant_docs, limit)
        f1_score = evaluate_f1_score(precision, recall)

        print(
            f" Scores Of The Results Based On Query: {query} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "retrieved": retrieved_documents[:limit],
            "relevant": list(relevant_docs)
        }

        total_precision += precision

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
        "total_precision": total_precision
    }
