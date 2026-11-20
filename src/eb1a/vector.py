"""Embedding retrieval. Hash vectors are an offline baseline only."""
import hashlib
import math
import re
from functools import lru_cache
from .schemas import RetrievalResult, SourceChunk
from .embedding_index import read_embedding_index

def hash_embed(text: str, dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()):
        index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]

embed = hash_embed

def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

def hash_vector_search(chunks, query, top_k=5):
    q = hash_embed(query)
    results = [RetrievalResult(chunk=c, score=cosine(q, hash_embed(c.text)), matched_terms=[]) for c in chunks]
    return sorted((r for r in results if r.score > 0), key=lambda r: (-r.score, r.chunk.chunk_id))[:top_k]

@lru_cache(maxsize=8)
def _sentence_model(model_name: str, local_files_only: bool = False):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic retrieval requires sentence-transformers. Install with: python -m pip install -e .[embeddings]") from exc
    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model '{model_name}': {exc}") from exc

def semantic_search(chunks, query, top_k=5, model_name="all-MiniLM-L6-v2", *, allow_download=True):
    if not chunks or not query.strip():
        return []
    model = _sentence_model(model_name, local_files_only=not allow_download)
    vectors = model.encode([query] + [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    q = vectors[0]
    results = [RetrievalResult(chunk=c, score=float(sum(a*b for a,b in zip(q, v))), matched_terms=[])
               for c, v in zip(chunks, vectors[1:])]
    return sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))[:top_k]

def indexed_semantic_search(chunks, query, index_path, top_k=5, model_name="all-MiniLM-L6-v2"):
    """Search a persisted embedding index; never recomputes corpus vectors."""
    payload = read_embedding_index(index_path, chunks, model_name)
    provider = _sentence_model(model_name, local_files_only=True)
    query_vector = provider.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    results = []
    for item in payload["items"]:
        chunk = by_id[item["chunk_id"]]
        score = float(sum(a * b for a, b in zip(query_vector, item["vector"])))
        results.append(RetrievalResult(chunk=chunk, score=score, matched_terms=[]))
    return sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))[:top_k]

def vector_search(chunks, query, top_k=5, *, model_name=None, allow_hash_baseline=False):
    # Kept as the legacy offline helper for compatibility. Semantic retrieval
    # is explicit via semantic_search(); production hybrid uses that path.
    return hash_vector_search(chunks, query, top_k)

def hybrid_search(chunks, query, top_k=5, *, model_name="all-MiniLM-L6-v2", allow_download=False):
    from .retrieve import bm25_search
    lexical = bm25_search(chunks, query, max(top_k, 10))
    semantic = semantic_search(chunks, query, max(top_k, 10), model_name, allow_download=allow_download)
    rankings = [lexical, semantic]
    scores, records = {}, {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, 1):
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0) + 1 / (60 + rank)
            records[result.chunk.chunk_id] = result
    return [records[k] for k, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]]

def indexed_hybrid_search(chunks, query, index_path, top_k=5, *, model_name="all-MiniLM-L6-v2", reranker_model=None):
    from .retrieve import bm25_search
    lexical = bm25_search(chunks, query, max(top_k, 10))
    semantic = indexed_semantic_search(chunks, query, index_path, max(top_k, 10), model_name)
    scores, records = {}, {}
    for ranking in (lexical, semantic):
        for rank, result in enumerate(ranking, 1):
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0) + 1 / (60 + rank)
            records[result.chunk.chunk_id] = result
    candidates = [records[k] for k, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:max(top_k, 10)]]
    if reranker_model:
        from .rerank import rerank
        return rerank(query, candidates, reranker_model, top_k)
    return candidates[:top_k]
