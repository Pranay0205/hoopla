from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


class MultiModalSearch:

    def __init__(self, model_name="clip-ViT-B-32"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])

        return image_embeddings[0]  # Return first element if it's a list


def verify_image_embedding(image_path: str):
    model = MultiModalSearch()

    image_embedding = model.embed_image(image_path)

    return image_embedding
