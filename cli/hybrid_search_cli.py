import argparse

from lib.hybrid_search import get_hybrid_search
from lib.llm_reranker import evaluate_results, llm_rerank_batch, llm_rerank_individual, re_rank
from lib.utils.constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT, DEFAULT_WEIGHT, SEARCH_LIMIT_MULTIPLIER
from lib.utils.hybrid_search_utils import normalize_scores
from lib.query_enhancer import query_enhancer


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

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Search content using Reciprocal Rank Fusion")

    rrf_parser.add_argument(
        "query", type=str, help="Query that needs to be searched")

    rrf_parser.add_argument("--k", type=int, default=DEFAULT_K,
                            help="K controls how much more weight we give to higher-ranked results vs. lower-ranked ones")

    rrf_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                            help="Limits the search results to set value")

    rrf_parser.add_argument("--enhance", type=str,
                            choices=["spell", "rewrite", "expand"], default="spell", help="Query enhancement method")

    rrf_parser.add_argument("--rerank-method", type=str,
                            choices=["individual", "batch", "cross_encoder"], default="individual", help="Query re rank method")

    rrf_parser.add_argument("--evaluate", action="store_true",
                            help="Evaluate the relevance of the results using LLM")

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

            enhanced_query = query_enhancer(args.enhance, args.query)

            print(
                f"Enhanced query ({args.enhance}): '{args.query}' -> {enhanced_query}\n")

            if not enhanced_query:
                raise ValueError("Failed to enhance the query")

            if args.rerank_method:
                results = hybrid_search.rrf_search(
                    enhanced_query, args.k, args.limit * SEARCH_LIMIT_MULTIPLIER)
            else:
                results = hybrid_search.rrf_search(
                    enhanced_query, args.k, args.limit)

            results = re_rank(enhanced_query, results,
                              args.rerank_method, args.limit)

            print(
                f"Reranking top {len(results)} results using {args.rerank_method} method...")
            print(
                f"Reciprocal Rank Fusion Results for '{enhanced_query}' (k={args.k})")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.get('title', 'Unknown')}")
                print(
                    f"\t\tCross Encoder Score: {result.get('cross_encoder_score', 0.00):.3f}/10")
                # print(f"\t\tRerank Score: {result['re_rank_score']:.3f}/10")
                print(f"\t\tRRF Score: {result.get('rrf_score', 0.00):.3f}")
                print(
                    f"\t\tBM25 Rank: {result.get('bm25_rank', 0)}, Semantic Rank: {result.get('semantic_rank', 0)}")
                print(f"\t\t{result.get('document', 'No document available')}")
                print()

            if args.evaluate:
                relevance_scores = evaluate_results(enhanced_query, results)

                print(f"Relevance Scores: {relevance_scores}\n")
                for i, score in enumerate(relevance_scores, 1):
                    print(f"{i}. {results[i-1]['title']}: {score}/3")


if __name__ == "__main__":
    main()
