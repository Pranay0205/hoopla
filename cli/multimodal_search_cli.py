import argparse
import os

from lib.multimodal_search import verify_image_embedding
from lib.utils.constants import PROJECT_ROOT


def main() -> None:

    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="generates verifies image embeddings")

    verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path of the image")

    args = parser.parse_args()

    match args.command:
        case "help":
            parser.print_help()

        case "verify_image_embedding":

            print("executing command verify image embedding")

            if args.image_path == "":
                raise ValueError("path cannot be empty")

            image_path = os.path.join(PROJECT_ROOT, args.image_path)

            image_embeddings = verify_image_embedding(image_path)

            if not image_embeddings:
                raise ValueError("no embeddings generated")

            print(f"Embedding shape: {image_embeddings.shape[0]} dimensions")


if __name__ == "__main__":
    main()
