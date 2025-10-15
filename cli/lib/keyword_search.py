import re
from .inverted_index import InvertedIndex
from .search_utils import (  # type: ignore
    BM25_B,
    BM25_K1,
    DEFAULT_SEARCH_LIMIT,
    stop_words_remover,
    tokenizer,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    if len(idx.docmap) == 0 or len(idx.index) == 0:
        raise ValueError("no index built yet!")

    query_tokens = tokenizer(query)
    query_tokens = stop_words_remover(query_tokens)

    matching_doc_ids = set()
    for token in query_tokens:
        if token in idx.index:
            matching_doc_ids.update(idx.index[token])

    result = []
    for id in sorted(matching_doc_ids):
        if id in idx.docmap:
            result.append(idx.docmap[id])
            if len(result) >= limit:
                break

    return result


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25search(query: str, limit=5):
    idx = InvertedIndex()
    idx.load()

    return idx.bm25_search(query, limit)
