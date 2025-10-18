from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("search text cannot be empty")

        embeddings = self.model.encode([text])

        return embeddings[0]


# Instantiate the model once and reuse it
semantic_model = SemanticSearch()


def verify_model():
    print(f"Model loaded: {semantic_model.model}")
    print(f"Max sequence length: {semantic_model.model.max_seq_length}")


def embed_text(text):
    embedding = semantic_model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
