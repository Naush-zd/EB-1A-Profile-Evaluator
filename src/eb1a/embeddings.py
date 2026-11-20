"""Provider interface and Sentence Transformers implementation."""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


_MODEL_CACHE = {}


def _load_model(model_name: str, local_files_only: bool = False):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install semantic retrieval with: python -m pip install -e .[embeddings]") from exc
    try:
        model = SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as exc:
        raise RuntimeError(f"embedding model '{model_name}' could not be loaded: {exc}") from exc
    _MODEL_CACHE[model_name] = model
    return model


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = _load_model(model_name)
        dimension_method = getattr(self._model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = self._model.get_sentence_embedding_dimension
        self.dimension = int(dimension_method())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]
