from .inverted_index import InvertedIndex
from .search_utils import (  # type: ignore
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
