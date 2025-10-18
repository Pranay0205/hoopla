#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Build the inverted index")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate text embedding")

    embed_text_parser.add_argument("text", type=str, help="Search query")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

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


if __name__ == "__main__":
    main()
