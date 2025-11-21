import argparse

from torch import Value, ge

from lib.llm_reranker import re_rank
from lib.query_enhancer import query_enhancer
from lib.hybrid_search import get_hybrid_search
from lib.rag_generator import generate_rag_response
from lib.utils.constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT, SEARCH_LIMIT_MULTIPLIER


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_parser.add_argument("--k", type=int, default=DEFAULT_K,
                            help="K controls how much more weight we give to higher-ranked results vs. lower-ranked ones")

    rag_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                            help="Limits the search results to set value")

    rag_parser.add_argument("--enhance", type=str,
                            choices=["spell", "rewrite", "expand"], default="rewrite", help="Query enhancement method")

    rag_parser.add_argument("--rerank-method", type=str,
                            choices=["individual", "batch", "cross_encoder"], default="individual", help="Query re-rank method")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            enhanced_query = query_enhancer(args.enhance, query)

            if not enhanced_query:
                raise ValueError("Failed to enhance the query")

            hybrid_search = get_hybrid_search()

            if args.rerank_method:
                results = hybrid_search.rrf_search(
                    enhanced_query, args.k, args.limit * SEARCH_LIMIT_MULTIPLIER)

            else:
                results = hybrid_search.rrf_search(
                    enhanced_query, args.k, args.limit)

            results = re_rank(enhanced_query, results,
                              args.rerank_method, args.limit)

            print("Search Results:\n")
            for res in results:
                print(f"  - {res["title"]}")

            response = generate_rag_response(query, results)

            print("RAG Response:")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
