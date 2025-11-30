import os
from urllib import response
from dotenv import load_dotenv  # type: ignore
from google import genai
from lib.utils.common_utils import rate_limit
from lib.utils.constants import GEMINI_MODEL
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = GEMINI_MODEL


def llm_image_describer(image: bytes, mime: str, query: str):
    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a 
                  movie database. 
                  Make sure to:
                - Synthesize visual and textual information
                - Focus on movie-specific details (actors, scenes, style, etc.)
                - Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        types.Part.from_text(text=query.strip())
    ]

    rate_limit()
    response = client.models.generate_content(model=model, contents=parts)

    if not response.text:
        raise ValueError("Model didn't had any response")

    return response
