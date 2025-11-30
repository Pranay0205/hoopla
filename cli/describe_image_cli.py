import argparse
import mimetypes
import os

from lib.llm_image_reader import llm_image_describer
from lib.utils.constants import PROJECT_ROOT


def main():

    parser = argparse.ArgumentParser(description="Describe Image CLI")

    parser.add_argument(
        "--image", type=str, help="Path to image file to describe")

    parser.add_argument(
        "--query", type=str, help="Question or query about the image")

    args = parser.parse_args()

    # Describe image
    print(f"Describing image {args.image} with query: {args.query}")

    mime, _ = mimetypes.guess_type(args.image)

    mime = mime or "image/jpeg"

    print(f"Type of the image: {mime}")

    file_path = os.path.join(PROJECT_ROOT, args.image)

    print(f"Looking for image in: {file_path}")

    if not os.path.exists(file_path):
        print("Path does not exist")

    with open(file_path, "rb") as f:
        image_data = f.read()

    response = llm_image_describer(image_data, mime, args.query)

    if not response.text:
        raise ValueError("No text in response from the model")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
