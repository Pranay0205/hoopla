from typing import Dict, List, Any
import json
import os

import numpy as np
from lib.search_utils import CACHE_DIR
from lib.semantic_search import SemanticSearch, semantic_chunk_text


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

        document_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_list.append(f"{doc['title']}: {doc['description']}")

        all_chunks = []
        chunk_metadata: List[Dict[str, Any]] = []

        for movie_idx, doc in enumerate(self.documents):
            if not doc['description']:
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

        document_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            document_list.append(f"{doc['title']}: {doc['description']}")

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):

            with open(self.chunk_embeddings_path, "rb") as f:
                self.chunk_embeddings = np.load(f)

            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
