from cgitb import text
from typing import List
import string
from .search_utils import (  # type: ignore
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)

from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenizer(query)
        title_tokens = tokenizer(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


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

    valid_tokens = stop_words_remover(valid_tokens)

    return valid_tokens


def stop_words_remover(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    valid_tokens = []
    for token in tokens:
        if token not in stopwords:
            valid_tokens.append(stemmer.stem(token))

    return valid_tokens
