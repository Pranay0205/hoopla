#!/usr/bin/env python3

import argparse

from lib.inverted_index import InvertedIndex
from lib.keyword_search import (
    search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            invertedidx = InvertedIndex()
            invertedidx.build()
            invertedidx.save()
            docs = invertedidx.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
