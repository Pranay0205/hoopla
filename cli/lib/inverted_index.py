from collections import defaultdict
import os
import pickle
from lib.keyword_search import tokenizer
from lib.search_utils import PROJECT_ROOT, load_movies


class InvertedIndex:
    index = defaultdict(set)
    docmap = {}

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
        cache_dir = os.path.join(PROJECT_ROOT, "cache")

        os.makedirs(cache_dir, exist_ok=True)

        print("saving index...")
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        print("saving index done ✅")

        print("saving docmap...")
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        print("saving docmap done ✅")
