import json
from dataclasses import dataclass
from pathlib import Path
from .corpus import chunk_text, read_jsonl, write_jsonl
from .ingest import clean_source_text, validate_corpus
from .embeddings import SentenceTransformerEmbeddingProvider
from .embedding_index import build_embedding_index


@dataclass(frozen=True)
class KnowledgeBaseManifest:
    source_url: str
    corpus_version: str
    chunk_count: int
    chunk_path: str
    validation_errors: list[str]

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            "corpus_version": self.corpus_version,
            "chunk_count": self.chunk_count,
            "chunk_path": self.chunk_path,
            "validation_errors": self.validation_errors,
        }


def build_knowledge_base(source_text: str, source_url: str, output: str | Path, manifest_path: str | Path | None = None, index_path: str | Path | None = None, embedding_model: str = "all-MiniLM-L6-v2") -> KnowledgeBaseManifest:
    cleaned = clean_source_text(source_text)
    chunks = chunk_text(cleaned, source_url, target_words=700, overlap_words=100)
    errors = validate_corpus(chunks)
    if errors:
        raise ValueError("knowledge base validation failed: " + "; ".join(errors))
    write_jsonl(chunks, output)
    manifest = KnowledgeBaseManifest(source_url, chunks[0].corpus_version, len(chunks), str(output), [])
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    if index_path:
        index_manifest = build_embedding_index(chunks, index_path, SentenceTransformerEmbeddingProvider(embedding_model))
        manifest = KnowledgeBaseManifest(source_url, chunks[0].corpus_version, len(chunks), str(output), [])
        if manifest_path:
            path.write_text(json.dumps({**manifest.to_dict(), "embedding_index": str(index_path), **index_manifest}, indent=2), encoding="utf-8")
    return manifest


def inspect_knowledge_base(path: str | Path) -> dict:
    chunks = read_jsonl(path)
    errors = validate_corpus(chunks)
    return {
        "chunk_count": len(chunks),
        "corpus_version": chunks[0].corpus_version if chunks else None,
        "source_urls": sorted({chunk.source_url for chunk in chunks}),
        "sections": sorted({chunk.section for chunk in chunks}),
        "headings": sorted({chunk.heading for chunk in chunks if chunk.heading}),
        "validation_errors": errors,
    }
