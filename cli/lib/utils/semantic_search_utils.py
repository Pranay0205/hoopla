
import re
import numpy as np
from lib.search_utils import format_search_result, load_movies
from lib.semantic_search import SemanticSearch


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


def search(query, limit=5):
    semantic_model = SemanticSearch()

    documents = load_movies()

    _ = semantic_model.load_or_create_embeddings(documents)

    output = semantic_model.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(output)} results:")
    print()

    for i, result in enumerate(output, 1):
        formatted = format_search_result(
            doc_id=result.get('id', ''),
            title=result['title'],
            document=result['description'],
            score=result['score']
        )
        print(f"{i}. {formatted['title']} (score: {formatted['score']:.2f})")
        print(f"   {formatted['document'][:100]}...")
        print()


def chunk_text(query, chunk_size, overlap):
    words = query.split()

    chunk_words = []
    chunks = []
    number = 1

    print(f"Chunking {len(query)} characters")
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


def semantic_chunk_text(query, max_chunk_size, overlap):
    query = query.strip()
    if len(query) == 0:
        return []

    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, query)

    chunked_sentences = []
    chunks = []

    if len(sentences) == 1 and not sentences[0].endswith(('.', '?', '!')):
        chunked_sentences.append(sentences)
        chunks.append(sentences)
        return chunks

    for sentence in sentences:
        chunked_sentences.append(sentence.strip())

        if len(chunked_sentences) == max_chunk_size:
            chunk = " ".join(chunked_sentences)

            if len(chunk.strip()) == 0:
                continue

            chunks.append(chunk)

            if overlap > 0:
                chunked_sentences = chunked_sentences[-overlap:]
            else:
                chunked_sentences = []

    if chunked_sentences:
        chunk = " ".join(chunked_sentences)
        chunks.append(chunk)

    return chunks
