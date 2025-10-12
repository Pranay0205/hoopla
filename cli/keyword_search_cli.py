#!/usr/bin/env python3

import argparse
import math

from lib.inverted_index import InvertedIndex, build_command
from lib.keyword_search import (
    bm25_idf_command,
    idf_command,
    search_command,
    tf_command,
    tf_idf_command,
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
        "idf", help="Get Inverse frequency of a term in the documents")
    idf_parser.add_argument("term", type=str, help="Search Term")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF of a term in the document")
    tf_idf_parser.add_argument("doc_id", type=str, help="Document Id")
    tf_idf_parser.add_argument("term", type=str, help="Search Term")

    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

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
                frequency = tf_command(args.doc_id, args.term)
                if frequency == 0:
                    print("0\n")
                else:
                    print(frequency, end="\n")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case "idf":
            try:
                idf = idf_command(args.term)
                print(
                    f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case "tfidf":
            try:
                tf_idf = tf_idf_command(args.doc_id, args.term)
                print(
                    f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case "bm25idf":
            try:

                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
