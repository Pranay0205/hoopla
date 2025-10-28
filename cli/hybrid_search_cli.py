import argparse

from lib.utils.hybrid_search_utils import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",  help="Normalize the scores of BM25")

    normalize_parser.add_argument(
        "scores", nargs='+', type=float, help="usage: normalize <values with space in between>.")

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()

        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")


if __name__ == "__main__":
    main()
