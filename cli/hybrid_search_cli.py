import argparse

from lib.hybrid_search import get_hybrid_search
from lib.utils.constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT, DEFAULT_WEIGHT
from lib.utils.hybrid_search_utils import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",  help="Normalize the scores of BM25")

    normalize_parser.add_argument(
        "scores", nargs='+', type=float, help="usage: normalize <values with space in between>.")

    weighted_search = subparsers.add_parser(
        "weighted-search", help="Search content using weighted search")

    weighted_search.add_argument(
        "query", type=str, help="Query that needs to be searched")

    weighted_search.add_argument(
        "--alpha", type=float, default=DEFAULT_WEIGHT,  help="Alpha value to decide to search semantically or keywordly")

    weighted_search.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                                 help="Limits the search results to set value")

    rrf_search = subparsers.add_parser(
        "rrf-search", help="Search content using Reciprocal Rank Fusion")

    rrf_search.add_argument(
        "query", type=str, help="Query that needs to be searched")

    rrf_search.add_argument("--k", type=int, default=DEFAULT_K,
                            help="K controls how much more weight we give to higher-ranked results vs. lower-ranked ones")

    rrf_search.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                            help="Limits the search results to set value")

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()

        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            hybrid_search = get_hybrid_search()

            results = hybrid_search.weighted_search(
                args.query, args.alpha, args.limit)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"   BM25: {result['keyword_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {result['document']}")
                print()

        case "rrf-search":
            hybrid_search = get_hybrid_search()

            results = hybrid_search.rrf_search(args.query, args.k, args.limit)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"\t\tRRF Score: {result["rrf_score"]:.3f}")
                print(
                    f"\t\tBM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
                print(f"\t\t{result['document']}")
                print()


if __name__ == "__main__":
    main()
