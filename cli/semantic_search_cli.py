#!/usr/bin/env python3

import argparse
from lib.semantic_search import search, semantic_chunk_text, verify_model, embed_query_text, embed_text, verify_embeddings, chunk_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Build the inverted index")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate text embedding")

    embed_text_parser.add_argument("text", type=str, help="Search query")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embeddings for the query")

    embed_query_parser.add_argument("query", type=str, help="Search query")

    semantic_search_parser = subparsers.add_parser(
        "search", help="Execute a semantic search to retrieve results")

    semantic_search_parser.add_argument(
        "query", type=str, help="The search query to process")
    semantic_search_parser.add_argument(
        "--limit", "-l", type=int, default=5, help="Specify the maximum number of results to return (default: 5)")

    chunk_parser = subparsers.add_parser(
        "chunk", help="Excute a chunked search")

    chunk_parser.add_argument(
        "query", type=str, help="The search query to process")

    chunk_parser.add_argument("--chunk-size", type=int, default=200,
                              help="Specify the chunk size to chunk the result (default = 200)")

    chunk_parser.add_argument("--overlap", type=int, default=2,
                              help="Specify the overlap size between chunks (default = 2)")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Excute a semantic chunking search")

    semantic_chunk_parser.add_argument(
        "query", type=str, help="The search query to process")

    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4,
                                       help="Specify the chunk size to chunk the result (default = 4)")

    semantic_chunk_parser.add_argument("--overlap", type=int, default=0,
                                       help="Specify the overlap size between chunks (default = 0)")

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            search(args.query, args.limit)

        case "chunk":
            chunk_text(args.query, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_text(args.query, args.max_chunk_size, args.overlap)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
