import json
import os

from dotenv import load_dotenv  # type: ignore
from google import genai

from lib.utils.common_utils import rate_limit
from lib.query_enhancer import query_enhancer
from lib.hybrid_search import get_hybrid_search
from lib.utils.constants import DEFAULT_K, GEMINI_MODEL, SEARCH_LIMIT_MULTIPLIER


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = GEMINI_MODEL


def _generate_llm_response(prompt: str) -> str:
    """Common function to generate LLM response with rate limiting."""
    rate_limit()
    response = client.models.generate_content(model=model, contents=prompt)

    if response.text:
        return response.text.strip()

    return "No response from LLM"


def _create_prompt(template: str, query: str, results: list[dict]) -> str:
    """Common function to create prompts with documents."""
    docs = json.dumps(results)
    return template.format(query=query, docs=docs)


def generate_rag_response(query: str, results: list[dict]) -> str:
    template = """Answer the question or provide information based on the provided documents.
                  This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                  Query: {query}

                  Documents:
                  {docs}

                  Provide a comprehensive answer that addresses the query:"""

    prompt = _create_prompt(template, query, results)
    return _generate_llm_response(prompt)


def llm_summarization(query: str, results: list[dict]) -> str:
    template = """
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {docs}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """

    prompt = _create_prompt(template, query, results)
    response = _generate_llm_response(prompt)

    if not response or response == "No response from LLM":
        raise ValueError("Failed to get response from LLM")

    return response


def llm_citation(query: str, results: list[dict]) -> str:

    template = """Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {docs}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""

    prompt = _create_prompt(template, query, results)

    response = _generate_llm_response(prompt)

    if not response or response == "No response from LLM":
        raise ValueError("Failed to get response from LLM")

    return response


def llm_question_and_answer(question: str, results: list[dict]) -> str:
    template = """Answer the user's question based on the provided movies that are available on Hoopla.
                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Question: {query}

                    Documents:
                    {docs}

                    Instructions:
                    - Answer questions directly and concisely
                    - Be casual and conversational
                    - Don't be cringe or hype-y
                    - Talk like a normal person would in a chat conversation

                    Answer:"""

    prompt = _create_prompt(template, question, results)

    response = _generate_llm_response(prompt)

    if not response or response == "No response from LLM":
        raise ValueError("Failed to get response from LLM")

    return response


def _search_and_display_results(query: str, k: int, limit: int, method: str = "individual") -> list[dict]:
    """Common function to search and display results."""
    print("\nSearching...")
    hybrid_search = get_hybrid_search()

    search_limit = limit * SEARCH_LIMIT_MULTIPLIER if method else limit
    results = hybrid_search.rrf_search(query, k, search_limit)

    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results, 1):
        print(f"  {i}. {res['title']}")

    return results


def _print_response(response: str):
    """Common function to print llm response."""
    print(response)


def llm_summarizer_command(query: str, limit: int):
    """Execute LLM Summarization pipeline with query"""
    print(f"Original query: {query}")
    results = _search_and_display_results(query, DEFAULT_K, limit)

    print("\nGenerating summary...")
    summary = llm_summarization(query, results)
    _print_response(summary)


def citation_command(query: str, limit: int):
    """Execute LLM Summarization pipeline with query"""
    print(f"Original query: {query}")
    results = _search_and_display_results(query, DEFAULT_K, limit)

    print("\nGenerating summary...")
    summary = llm_citation(query, results)
    _print_response(summary)


def question_command(question: str, limit: int):
    """Execute LLM Question and Answer pipeline with question"""
    print(f"Original Question: {question}")
    results = _search_and_display_results(question, DEFAULT_K, limit)

    print("\n Generating Answer...")
    answer = llm_question_and_answer(question=question, results=results)

    _print_response(answer)


def rag_command(query: str, enhance: str, method: str, k: int, limit: int):
    """Execute RAG pipeline with enhanced query processing and summarization."""
    print(f"Original query: {query}")
    enhanced_query = query_enhancer(enhance, query)

    if not enhanced_query:
        raise ValueError("Failed to enhance the query")

    if enhanced_query != query:
        print(f"Enhanced query: {enhanced_query}")

    results = _search_and_display_results(enhanced_query, k, limit, method)

    print("\nGenerating summary...")
    response = generate_rag_response(query, results)
    _print_response(response)

    return response
