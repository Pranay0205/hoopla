from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


class MultiModalSearch:

    def __init__(self, model_name="jinaai/jina-clip-v2"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True)

    def embed_image(self, image_path: str):

        image = Image.open(image_path)
        image_embeddings = self.model.encode_image([image])

        return image_embeddings


def verify_image_embedding(image_path: str):
    model = MultiModalSearch()

    image_embedding = model.embed_image(image_path)
