import os
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

    def search(self, query, limit):

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
            output.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })

        return output


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


def embed_query_text(query):
    semantic_model = SemanticSearch()

    embedding = semantic_model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def truncate_text(text, length=100):
    return text[:length] + "..." if len(text) > length else text


def search(query, limit=5):
    semantic_model = SemanticSearch()

    documents = load_movies()

    _ = semantic_model.load_or_create_embeddings(documents)

    output = semantic_model.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(output)} results:")
    print()

    for i, result in enumerate(output):
        print(f"{i + 1}. {result["title"]} (score: {result["score"]:.2f})")
        print(f"{truncate_text(result["description"])}\n")


def chunk_text(query, chunk_size, overlap):
    words = query.split()

    print(f"Chunking {len(query)} characters")

    chunk_words = []
    chunks = []
    number = 1
    for _, word in enumerate(words):
        chunk_words.append(word)

        if len(chunk_words) == chunk_size:
            chunk = " ".join(chunk_words)
            print(f"{number}. {chunk}")
            chunks.append(chunk)
            number += 1

            if overlap > 0:
                chunk_words = chunk_words[-overlap:]
            else:
                chunk_words = []

    if chunk_words:
        chunk = " ".join(chunk_words)
        print(f"{number}. {chunk}")
        chunks.append(chunk)

    return chunks
