#!/usr/bin/env python3

import argparse
from lib.semantic_search import search, verify_model, embed_query_text, embed_text, verify_embeddings


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
