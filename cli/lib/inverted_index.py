from audioop import reverse
from collections import defaultdict
import math
import os
import pickle
from textwrap import indent
import token
from typing import Counter
from lib.search_utils import BM25_B, BM25_K1, stop_words_remover, tokenizer, CACHE_DIR, load_movies


class InvertedIndex:

    def __init__(self) -> None:
        # index = "term": {"1", "2", "3", "4"}
        self.index = defaultdict(set)

        # docmap = "1" : {"id": "1", "title": "movie_title", "description": "<description>", ...}
        self.docmap: dict[int, dict] = {}

        # term_freq = "1" : Counter({"term_1": 4 -> (count), "term_2": 6 ...} ...)
        self.term_frequencies = defaultdict(Counter)

        self.doc_length: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenizer(text)
        tokens = stop_words_remover(tokens)
        self.doc_length[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:

        number_of_docs = len(self.doc_length)
        total_doc_length = 0
        for doclength in self.doc_length.values():
            total_doc_length += doclength

        if number_of_docs == 0 or total_doc_length == 0:
            return 0.0

        return total_doc_length / number_of_docs

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

        print("saving document lengths...")
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_length, f)

    def load(self):
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(
                f"Docmap file not found: {self.docmap_path}")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        if not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError(
                f"Term Frequencies file not found: {self.term_frequencies_path}")

        if not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError(
                f"Document Length file not found: {self.doc_lengths_path}")

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_length = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:

        terms = tokenizer(term)
        terms = stop_words_remover(terms)

        if len(terms) > 1:
            raise Exception("term should be singular")

        if not terms:
            return 0

        return self.term_frequencies[doc_id][terms[0]]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.term_frequencies)

        terms = tokenizer(term)
        terms = stop_words_remover(terms)

        if not terms:
            term_doc_count = 0
        else:
            term_doc_count = len(self.index[terms[0]])

        idf = math.log((doc_count + 1) / (term_doc_count + 1))

        return idf

    def get_tf_idf(self, doc_id: int,  term: str) -> float:

        tf = self.get_tf(doc_id, term)

        terms = tokenizer(term)
        terms = stop_words_remover(terms)

        if not terms:
            doc_freq = 0
        else:
            doc_freq = len(self.index[terms[0]])

        doc_count = len(self.docmap)

        idf = math.log((doc_count + 1) / (doc_freq + 1))

        return tf * idf

    def get_bm25_idf(self, term: str) -> float:

        doc_count = len(self.docmap)
        terms = tokenizer(term)
        terms = stop_words_remover(terms)

        if not terms:
            doc_freq = 0
        else:
            doc_freq = len(self.index[terms[0]])

        bm25_idf = math.log(
            (doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)

        length_norm = 1 - b + b * \
            (self.doc_length[doc_id] / self.__get_avg_doc_length())

        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int):
        terms = tokenizer(query)
        terms = stop_words_remover(terms)
        bm25_scores: dict[int, float] = {}

        for term in terms:
            for doc_id in self.index[term]:
                if doc_id not in bm25_scores:
                    bm25_scores[doc_id] = self.bm25(doc_id, term)
                else:
                    bm25_scores[doc_id] += self.bm25(doc_id, term)

        sorted_bm25_score = sorted(bm25_scores.items(),
                                   key=lambda item: item[1], reverse=True)

        for i, item in enumerate(sorted_bm25_score):
            if i < limit:
                document = self.docmap[item[0]]
                print(
                    f"{i + 1}. ({document["id"]}) {document["title"]} - Score: {item[1]:.2f}")


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
