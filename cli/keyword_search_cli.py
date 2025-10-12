#!/usr/bin/env python3

import argparse
import math

from lib.inverted_index import InvertedIndex, build_command
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

    tf_parser = subparsers.add_parser(
        "tf", help="Get frequency of a term in the document")
    tf_parser.add_argument("doc_id", type=str, help="Document Id")
    tf_parser.add_argument("term", type=str, help="Search Term")

    idf_parser = subparsers.add_parser(
        "idf", help="Get Inverse frequency of a term in the document")
    idf_parser.add_argument("term", type=str, help="Search Term")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()

        case "search":
            try:
                print("Searching for:", args.query)
                results = search_command(args.query)
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['title']}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the index.")
        case "tf":
            try:
                idx = InvertedIndex()
                idx.load()
                frequency = idx.get_tf(args.doc_id, args.term)
                if frequency == 0:
                    print("0\n")
                else:
                    print(frequency, end="\n")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case "idf":
            try:
                idx = InvertedIndex()
                idx.load()
                idf = idx.get_idf(args.term)

                print(
                    f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
