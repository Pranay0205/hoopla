import json
import os
import time

from dotenv import load_dotenv  # type: ignore
from google import genai
from torch import Value

from lib.utils.constants import DEFAULT_SEARCH_LIMIT


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"


def llm_rerank_individual(query: str, results: list, limit) -> list:

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

    return sorted_results[:limit]


def llm_rerank_batch(query: str, results: list, limit: int) -> list:

    id_to_movie = {result["id"]: result for result in results}
    movies = [
        f"Id: {r["id"]}, Title: {r["title"]}, Document:{r["document"]}" for r in results]

    doc_list_str = "\n".join(movies)

    prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else not even the quotations or backticks. For example:

                [75, 12, 34, 2, 1]
                """
    response = client.models.generate_content(model=model, contents=prompt)
    if response.text:
        ranked_ids = json.loads(response.text.strip())

    reranked_movies = []
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in id_to_movie:
            result = id_to_movie[doc_id].copy()
            result["re_rank_score"] = rank
            reranked_movies.append(result)

    sorted_results = sorted(
        reranked_movies, key=lambda x: x.get("re_rank_score", 0), reverse=True)

    return sorted_results[:limit]


def re_rank(query: str, document: list[dict], method: str = "batch", limit: int = DEFAULT_SEARCH_LIMIT):

    results = []
    match method:
        case "individually":
            results = llm_rerank_individual(query, document, limit)
        case "batch":
            results = llm_rerank_batch(query, document, limit)
        case _:
            raise ValueError("unknown method used")

    return results
