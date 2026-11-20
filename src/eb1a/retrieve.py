import re
import math
from .schemas import RetrievalResult, SourceChunk


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())


def search(chunks: list[SourceChunk], query: str, top_k: int = 5) -> list[RetrievalResult]:
    query_terms = set(_terms(query))
    if not query_terms:
        return []
    results = []
    for chunk in chunks:
        chunk_terms = set(_terms(chunk.text))
        matched = sorted(query_terms & chunk_terms)
        if matched:
            score = len(matched) / max(len(query_terms), 1)
            results.append(RetrievalResult(chunk=chunk, score=score, matched_terms=matched))
    return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))[:top_k]


def bm25_search(chunks: list[SourceChunk], query: str, top_k: int = 5) -> list[RetrievalResult]:
    """Small dependency-free BM25 implementation for the corpus baseline."""
    query_terms = _terms(query)
    if not query_terms or not chunks:
        return []
    docs = [_terms(chunk.text) for chunk in chunks]
    avgdl = sum(len(doc) for doc in docs) / len(docs)
    k1, b = 1.5, 0.75
    results = []
    for chunk, doc in zip(chunks, docs):
        frequencies = {term: doc.count(term) for term in set(query_terms)}
        score = 0.0
        matched = []
        for term, tf in frequencies.items():
            if not tf:
                continue
            df = sum(term in item for item in docs)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(doc) / max(avgdl, 1)))
            matched.append(term)
        if matched:
            results.append(RetrievalResult(chunk=chunk, score=score, matched_terms=sorted(matched)))
    return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))[:top_k]
