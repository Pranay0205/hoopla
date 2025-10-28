import json
import os
import string
from typing import Any
from nltk.stem import PorterStemmer  # type: ignore

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5
BM25_B = 0.75
DOCUMENT_PREVIEW_LIMIT = 100
DEFAULT_SEARCH_LIMIT = 5


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def processed_text(text: str) -> str:
    text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)
    return text


def tokenizer(text: str) -> list[str]:
    text = processed_text(text)
    list_text = text.split()
    valid_tokens = []
    for token in list_text:
        if token:
            valid_tokens.append(token)

    return valid_tokens


def stop_words_remover(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    valid_tokens = []
    for token in tokens:
        if token not in stopwords:
            valid_tokens.append(stemmer.stem(token))

    return valid_tokens


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document[:DOCUMENT_PREVIEW_LIMIT],
        "score": round(score, 3),
        "metadata": metadata if metadata else {},
    }
