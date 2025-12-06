from unittest import result
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.utils.constants import DEFAULT_SEARCH_LIMIT
from lib.utils.math_utils import cosine_similarity
from lib.utils.search_utils import format_search_result, load_movies


class MultiModalSearch:

    def __init__(self, model_name="clip-ViT-B-32", document=[]):
        self.model_name = model_name
        self.model = SentenceTransformer(
            self.model_name, model_kwargs={'use_fast': False})
        self.document = document
        self.texts = [
            f"{doc['title']}: {doc['description']}" for doc in self.document]
        self.text_embeddings = self.model.encode(
            self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image], show_progress_bar=True)

        return image_embeddings[0]

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        result = []
        for i, embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(embedding, image_embedding)

            result.append(format_search_result(
                self.document[i]["id"], self.document[i]["title"], score=similarity, document=self.document[i]["description"]))

        result.sort(key=lambda x: x["score"], reverse=True)

        return result[:DEFAULT_SEARCH_LIMIT]


def verify_image_embedding(image_path: str):
    model = MultiModalSearch()

    image_embedding = model.embed_image(image_path)

    return image_embedding


def image_search_command(image_path: str) -> list[dict]:

    documents = load_movies()

    model = MultiModalSearch(document=documents)

    results = model.search_with_image(image_path)

    return results
