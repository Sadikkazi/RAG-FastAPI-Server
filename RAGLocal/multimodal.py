from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from urllib.request import urlopen
import numpy as np

class ImageRAG:
    def __init__(self, model_name: str):
        # 1) Definir dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 2) Cargar modelo y processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _load_image(self, image):
        if isinstance(image, str) and image.startswith("http"):
            return Image.open(urlopen(image)).convert("RGB")
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError("La imagen debe ser path, URL o PIL.Image.")

    def get_embeddings(self, image) -> np.ndarray:
        """
        Retorna el embedding L2-normalizado (512-D) de una imagen.
        """
        # Carga
        img = self._load_image(image)
        # Preprocesado y envío a dispositivo
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        # Inferencia
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        # Normalización
        norms = image_features.norm(p=2, dim=-1, keepdim=True)
        embedding_norm = image_features / norms
        # Convertir a numpy y retornar (batch_size=1, 512)
        return embedding_norm.cpu().numpy()[0]



if __name__ == "__main__":
    from rag import RAGLocal
    model_name = "openai/clip-vit-base-patch32"
    image_rag = ImageRAG(model_name)

    rag = RAGLocal(dbname="mi_rag", user="user", password="password", host="localhost", rag_multimodal=image_rag)

    # Crea tabla + índice
    #rag.create_image_index("images2", "path")

    # Indexa una imagen
    #id_leon = rag.add_image("images2", "path", "leonA.jpg")

    #id_leon2 = rag.add_image("images2", "path", "leonC.jpg")

    #id_leop = rag.add_image("images2", "path", "leopardo.jpg")

    #man_id = rag.add_image("images2", "path", "man.jpg")

    query_image = "leonA.jpg"

    print(f"Query: {query_image}")

    resultados = rag.query_image("images2", "path", query_image, top_k=3)
    for r in resultados:
        print(f"ID: {r['id']}, Path: {r['content']}, Score: {r['score']:.3f}")