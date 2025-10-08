from collections import defaultdict
import os
import pickle
from lib.keyword_search import tokenizer
from lib.search_utils import CACHE_DIR, PROJECT_ROOT, load_movies


class InvertedIndex:

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        self.docmap[doc_id] = text

        tokens = tokenizer(text)

        for token in tokens:
            self.index[token].add(doc_id)

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

    def load(self):
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(
                f"Docmap file not found: {self.docmap_path}")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
