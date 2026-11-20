"""Provider interface and Sentence Transformers implementation."""
from abc import ABC, abstractmethod
from functools import lru_cache


class EmbeddingProvider(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install semantic retrieval with: python -m pip install -e .[embeddings]") from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"embedding model '{model_name}' could not be loaded: {exc}") from exc


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = _load_model(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]
