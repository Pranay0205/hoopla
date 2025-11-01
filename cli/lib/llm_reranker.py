import os
import time

from dotenv import load_dotenv  # type: ignore
from google import genai

from lib.utils.constants import DEFAULT_SEARCH_LIMIT  # type: ignore

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"


def rerank_results(query: str, results: list) -> list:

    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

                  Query: "{query}"
                  Movie: {doc.get("title", "")} - {doc.get("document", "")}

                  Consider:
                  - Direct relevance to query
                  - User intent (what they're looking for)
                  - Content appropriateness

                  Rate 0-10 (10 = perfect match).
                  Give me ONLY the number in your response, no other text or explanation.

                  Score:"""
        response = client.models.generate_content(model=model, contents=prompt)
        if response.text:
            try:
                doc["re_rank_score"] = int(response.text.strip())
            except ValueError:
                print(
                    f"Warning: Invalid score '{response.text}' for {doc.get('title', 'Unknown')}")
                doc["re_rank_score"] = 0

        time.sleep(3)

        sorted_results = sorted(
            results, key=lambda x: x.get("re_rank_score", 0), reverse=True)

    return sorted_results
