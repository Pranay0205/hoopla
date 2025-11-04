import argparse

from lib.hybrid_search import get_hybrid_search
from lib.query_enhancer import query_enhancer
from lib.utils.evaluation_utils import load_evaluation_dataset


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_evaluation_dataset()

    hybrid_search = get_hybrid_search()

    total_retrieved = []
    for test_case in golden_dataset:
        enhanced_query = query_enhancer("expand", test_case["query"])
        print(
            f"Enhanced query (expand): '{test_case["query"]}' -> {enhanced_query}\n")

        if not enhanced_query:
            raise ValueError("Failed to enhance the query")

        results = hybrid_search.rrf_search(
            enhanced_query, 60, args.limit)

        total_retrieved.append(results)

    for test_case, results in zip(golden_dataset, total_retrieved):
        relevant_docs = test_case["relevant_docs"]

        relevant_retrieved = sum(1 for rel in relevant_docs if rel in results)

        precision = relevant_retrieved / len(results)

        print("k = 60")

        print(f"- Query: {test_case["query"]}")
        print(f"\t- Precision@6: {precision:.4f}")
        print(f"\t- Retrieved: {",".join(results)}")
        print(f"\t- Relevant: {",".join(relevant_docs)}")


if __name__ == "__main__":
    main()
