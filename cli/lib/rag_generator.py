import json
import os

from dotenv import load_dotenv  # type: ignore
from google import genai

from lib.llm_reranker import rate_limit
from lib.utils.constants import GEMINI_MODEL


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = GEMINI_MODEL


def generate_rag_response(query: str, results: list[dict]) -> str:

    docs = json.dumps(results)

    prompt = f"""Answer the question or provide information based on the provided documents. 
                  This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                  Query: {query}

                  Documents:
                  {docs}

                  Provide a comprehensive answer that addresses the query:"""

    rate_limit()
    response = client.models.generate_content(model=model, contents=prompt)

    if response.text:

        rag_response = response.text.strip()

        return rag_response

    return "No response from LLM"
