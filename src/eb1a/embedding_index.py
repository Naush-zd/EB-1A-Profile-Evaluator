"""Persistent JSON vector index for the USCIS corpus."""
import hashlib
import json
from pathlib import Path
from .embeddings import EmbeddingProvider
from .schemas import SourceChunk


def index_version(chunks: list[SourceChunk], model_name: str) -> str:
    value = model_name + "|" + "|".join(f"{c.chunk_id}:{c.content_hash}" for c in chunks)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def build_embedding_index(chunks: list[SourceChunk], output: str | Path, provider: EmbeddingProvider) -> dict:
    vectors = provider.embed([chunk.text for chunk in chunks])
    version = index_version(chunks, provider.model_name)
    payload = {
        "index_version": version,
        "embedding_model": provider.model_name,
        "dimension": provider.dimension,
        "corpus_version": chunks[0].corpus_version if chunks else "empty",
        "items": [{"chunk_id": chunk.chunk_id, "content_hash": chunk.content_hash, "vector": vector}
                  for chunk, vector in zip(chunks, vectors)],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {key: payload[key] for key in ("index_version", "embedding_model", "dimension", "corpus_version")}


def read_embedding_index(path: str | Path, chunks: list[SourceChunk], model_name: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = index_version(chunks, model_name)
    if payload.get("index_version") != expected or payload.get("corpus_version") != (chunks[0].corpus_version if chunks else "empty"):
        raise ValueError("embedding index is stale for this corpus or model")
    return payload
