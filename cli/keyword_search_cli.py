#!/usr/bin/env python3

import argparse
import math

from lib.inverted_index import InvertedIndex, build_command
from lib.utils.keyword_search_utils import (
    bm25_idf_command,
    bm25_tf_command,
    bm25search,
    idf_command,
    search_command,
    tf_command,
    tf_idf_command,
)
from lib.utils.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # TF
    tf_parser = subparsers.add_parser(
        "tf", help="Get frequency of a term in the document")
    tf_parser.add_argument("doc_id", type=int, help="Document Id")
    tf_parser.add_argument("term", type=str, help="Search Term")

    # IDF
    idf_parser = subparsers.add_parser(
        "idf", help="Get Inverse frequency of a term in the documents")
    idf_parser.add_argument("term", type=str, help="Search Term")

    # TF-IDF
    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF of a term in the document")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document Id")
    tf_idf_parser.add_argument("term", type=str, help="Search Term")

    # BM25 IDF
    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    # BM25 TF
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument(
        "b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # BM25 SEARCH
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", "-l", type=int, default=5,  help="Number of results to return (default: 5)")

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

        case "bm25tf":
            try:

                bm25tf = bm25_tf_command(
                    args.doc_id, args.term, args.k1, args.b)

                print(
                    f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case "bm25search":
            try:
                bm25search(args.query, args.limit)

            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Please run 'build' command first to create the term frequencies.")

        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
