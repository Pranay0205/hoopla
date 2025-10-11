from collections import defaultdict
import os
import pickle
from typing import Counter
from lib.search_utils import stop_words_remover, tokenizer, CACHE_DIR, load_movies


class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: str, text: str) -> None:
        tokens = tokenizer(text)
        tokens = stop_words_remover(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index[term]

        sorted_ids = sorted(doc_ids)

        return sorted_ids

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie['id']
            self.docmap[doc_id] = movie
            input_text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(doc_id, input_text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("saving index...")
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        print("saving docmap...")
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        print("saving term frequencies...")
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(
                f"Docmap file not found: {self.docmap_path}")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        if not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError(
                f"Term Frequencies file not found: {self.term_frequencies_path}")

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: str, term: str) -> int:

        terms = tokenizer(term)
        terms = stop_words_remover(terms)

        if len(terms) > 1:
            raise Exception("term should be singular")

        if not terms:
            return 0

        return self.term_frequencies[int(doc_id)][terms[0]]


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
