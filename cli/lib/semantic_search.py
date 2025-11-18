
import os
from sentence_transformers import SentenceTransformer
import numpy as np

from lib.utils.constants import CACHE_DIR
from lib.utils.search_utils import format_search_result
from lib.utils.math_utils import cosine_similarity


class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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

        print(
            f"Generated embedding for text: {text[:30]}... -> {embeddings[0][:5]}...")

        return embeddings[0]

    def search(self, query: str, limit: int) -> list:

        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first.")

        query_emb = self.generate_embedding(query)

        results = []
        for i, embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_emb, embedding)
            results.append((score, self.documents[i]))

        results.sort(key=lambda item: item[0], reverse=True)

        output = []
        for i, (score, doc) in enumerate(results[:limit]):
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score
            )
            output.append(formatted_result)

        return output
