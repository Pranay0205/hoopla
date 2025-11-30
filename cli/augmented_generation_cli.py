import argparse


from lib.augment_generation import citation_command, llm_summarizer_command, question_command, rag_command
from lib.utils.constants import DEFAULT_K, DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # RAG Command Parser
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

    # LLM Summarize Parser
    llm_summarize_parser = subparsers.add_parser(
        "summarize", help="Performs summarization of results"
    )

    llm_summarize_parser.add_argument(
        "query", type=str, help="Search query for RAG")

    llm_summarize_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limits the search results to set value")

    # Citation Parser
    citation_parser = subparsers.add_parser(
        "citations", help="Performs summarization with citation of results")

    citation_parser.add_argument(
        "query", type=str, help="Search query for RAG")

    citation_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limits the search results to set value")

    # Question Parser
    question_parser = subparsers.add_parser(
        "question", help="Ask question to LLM to get answer")

    question_parser.add_argument(
        "query", type=str, help="Question to be asked")

    question_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limits the search results to set value")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            rag_command(query, args.enhance,
                        args.rerank_method, args.k, args.limit)
        case "summarize":
            query = args.query

            llm_summarizer_command(query, args.limit)

        case "citations":
            query = args.query

            citation_command(query, args.limit)

        case "question":
            query = args.query

            question_command(query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
