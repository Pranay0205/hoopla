import os

from dotenv import load_dotenv  # type: ignore
from google import genai  # type: ignore

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"


def query_enhancer(method: str, query: str) -> str | None:

    if not query.strip():
        raise ValueError("Query cannot be empty or whitespace.")
    if method not in {"spell", "rewrite", "expand"}:
        raise ValueError(
            f"Unsupported method: {method}. Choose 'spell' or 'rewrite' or 'expand'.")
    print(f"Enhancing query using method '{method}': {query}")

    match method:
        case "spell":

            prompt = f"""Fix any spelling errors in this movie search query.

                          Only correct obvious typos. Don't change correctly spelled words.

                          Query: "{query}"

                          If no errors, return the original query.
                          Corrected:"""

        case "rewrite":
            prompt = f"""Rewrite this movie search query to be more specific and searchable.

                          Original: "{query}"

                          Consider:
                          - Common movie knowledge (famous actors, popular films)
                          - Genre conventions (horror = scary, animation = cartoon)
                          - Keep it concise (under 10 words)
                          - It should be a google style search query that's very specific
                          - Don't use boolean logic

                          Examples:

                          - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                          - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                          - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                          Rewritten query:"""
        case "expand":
            prompt = f"""Expand this movie search query with related terms.

                          Add synonyms and related concepts that might appear in movie descriptions.
                          Keep expansions relevant and focused.
                          This will be appended to the original query.

                          Examples:

                          - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                          - "action movie with bear" -> "action thriller bear chase fight adventure"
                          - "comedy with bear" -> "comedy funny bear humor lighthearted"

                          Query: "{query}"
                          """
        case _:
            return query

    response = client.models.generate_content(model=model, contents=prompt)

    return response.text
