"""Optional neural cross-encoder reranking."""
from functools import lru_cache
from .schemas import RetrievalResult


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Reranking requires sentence-transformers") from exc
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(f"reranker model '{model_name}' could not be loaded: {exc}") from exc


def rerank(query: str, results: list[RetrievalResult], model_name: str, top_k: int) -> list[RetrievalResult]:
    if not results:
        return []
    model = _load_reranker(model_name)
    scores = model.predict([(query, item.chunk.text) for item in results], show_progress_bar=False)
    ranked = [item.model_copy(update={"score": float(score)}) for item, score in zip(results, scores)]
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
