from typing import Dict, List, Any
import json
import os

import numpy as np

from lib.semantic_search import SemanticSearch
from lib.utils.constants import CACHE_DIR, DEFAULT_SEARCH_LIMIT
from lib.utils.search_utils import format_search_result, load_movies
from lib.utils.semantic_search_utils import cosine_similarity, semantic_chunk_text


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:

        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(
            CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(
            CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents

        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata: List[Dict[str, Any]] = []

        for movie_idx, doc in enumerate(documents):
            text = doc.get('description', '')
            if not text.strip():
                continue

            chunks = semantic_chunk_text(doc['description'],
                                         max_chunk_size=4, overlap=1)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        with open(self.chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata,
                       "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):

            with open(self.chunk_embeddings_path, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):

        query_embeddings = self.generate_embedding(query)

        chunk_scores = []

        if self.chunk_embeddings is None:
            raise ValueError(
                "Chunk embeddings are not loaded. Please build or load embeddings before searching.")

        if self.chunk_metadata is None:
            raise ValueError(
                "Chunk metadata is not loaded. Please build or load embeddings before searching.")

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(
                chunk_embedding, query_embeddings)
            chunk_scores.append(
                {"chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                 "movie_idx": self.chunk_metadata[i]["movie_idx"],
                 "score": similarity_score
                 }
            )

        movies_scores: dict[int, float] = {}

        for chunk_score in chunk_scores:
            if chunk_score["movie_idx"] not in movies_scores or chunk_score["score"] > movies_scores[chunk_score["movie_idx"]]:
                movies_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        sorted_movies_scores = sorted(
            movies_scores.items(), key=lambda x: x[1], reverse=True)

        final_list = []
        for mv_score in sorted_movies_scores[:limit]:
            movie = self.documents[mv_score[0]]
            formatted = format_search_result(
                doc_id=movie['id'],
                title=movie['title'],
                document=movie['description'],
                score=mv_score[1]
            )
            final_list.append(formatted)

        return final_list


def search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    chunked_sem_model = ChunkedSemanticSearch()

    documents = load_movies()

    chunked_sem_model.load_or_create_chunk_embeddings(documents)

    results = chunked_sem_model.search_chunks(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


def embed_chunks():
    chunked_sem_model = ChunkedSemanticSearch()

    documents = load_movies()

    embeddings = chunked_sem_model.load_or_create_chunk_embeddings(
        documents)

    print(f"Generated {len(embeddings)} chunked embeddings")
