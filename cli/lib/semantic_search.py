import array
import os
from typing import Any, DefaultDict
from sentence_transformers import SentenceTransformer
import numpy as np

from lib.search_utils import CACHE_DIR, load_movies


class SemanticSearch:

    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, dict] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def build_embeddings(self, documents):
        self.documents = documents

        document_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(
            document_list, show_progress_bar=True)

        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        document_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_list.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(self.documents)

    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("search text cannot be empty")

        embeddings = self.model.encode([text])

        return embeddings[0]


def verify_model():
    semantic_model = SemanticSearch()
    print(f"Model loaded: {semantic_model.model}")
    print(f"Max sequence length: {semantic_model.model.max_seq_length}")


def embed_text(text):
    semantic_model = SemanticSearch()
    embedding = semantic_model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_model = SemanticSearch()

    documents = load_movies()

    embeddings = semantic_model.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
