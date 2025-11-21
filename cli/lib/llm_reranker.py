import json
import os
import time

from dotenv import load_dotenv  # type: ignore
from google import genai

from lib.utils.constants import DEFAULT_SEARCH_LIMIT, GEMINI_MODEL
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = GEMINI_MODEL


def llm_rerank_individual(query: str, documents: list, limit) -> list:

    for doc in documents:
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
            documents, key=lambda x: x.get("re_rank_score", 0), reverse=True)

    return sorted_results[:limit]


def llm_rerank_batch(query: str, documents: list, limit: int) -> list:

    id_to_movie = {result["id"]: result for result in documents}
    movies = [
        f"Id: {r["id"]}, Title: {r["title"]}, Document:{r["document"]}" for r in documents]

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


def llm_rerank_cross_encoder(query: str, documents: list[dict], limit: int) -> list:

    pairs = []
    for doc in documents:
        pairs.append(
            [query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):

        doc["cross_encoder_score"] = score

    sorted_documents = sorted(
        documents, key=lambda x: x["cross_encoder_score"], reverse=True)

    return sorted_documents[:limit]


def re_rank(query: str, documents: list[dict], method: str = "batch", limit: int = DEFAULT_SEARCH_LIMIT):

    results = []
    match method:
        case "individual":
            results = llm_rerank_individual(query, documents, limit)
        case "batch":
            results = llm_rerank_batch(query, documents, limit)
        case "cross_encoder":
            results = llm_rerank_cross_encoder(query, documents, limit)
        case _:
            raise ValueError("unknown method used")

    return results


def evaluate_results(query: str, results: list[dict]) -> list[int]:

    formatted_results = [
        f'"{res["title"]} - {res["document"]}"' for res in results]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers out than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model=model, contents=prompt)
    if response.text:
        try:
            scores = json.loads(response.text.strip())
        except json.JSONDecodeError:
            print("Warning: Failed to decode evaluation scores")

    return scores
