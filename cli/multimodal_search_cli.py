import argparse
import enum
import os

from lib.multimodal_search import image_search_command, verify_image_embedding
from lib.utils.constants import PROJECT_ROOT


def main() -> None:

    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="generates verifies image embeddings")

    verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path of the image")

    image_search = subparsers.add_parser(
        "image_search", help="search with image or description")

    image_search.add_argument("image_path", type=str, help="Path of the image")

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()

        case "verify_image_embedding":

            print("executing command verify image embedding...")

            if args.image_path == "":
                raise ValueError("path cannot be empty")

            image_path = os.path.join(PROJECT_ROOT, args.image_path)

            image_embeddings = verify_image_embedding(image_path)

            if image_embeddings is None or image_embeddings.size == 0:
                raise ValueError("no embeddings generated")

            print(f"Embedding shape: {image_embeddings.shape[0]} dimensions")

        case "image_search":

            print("excuting command for image search...")

            if args.image_path == "":
                raise ValueError("path cannot be empty")

            image_path = os.path.join(PROJECT_ROOT, args.image_path)

            results = image_search_command(image_path)

            if results is None or len(results) == 0:
                raise ValueError("no results found")

            for index, result in enumerate(results, 1):
                print(
                    f"{index}. {result["title"]} (similarity: {result["score"]:.3f})")
                print(f"  {result["document"]}...")


if __name__ == "__main__":
    main()
