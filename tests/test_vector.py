from eb1a.corpus import chunk_text
from eb1a.vector import embed, hybrid_search, vector_search, hash_vector_search


def test_hash_embeddings_are_deterministic():
    assert embed("awards and prizes") == embed("awards and prizes")


def test_hash_baseline_returns_results():
    chunks = chunk_text("Awards and prizes. Membership requires achievement.", "u", "t")
    assert vector_search(chunks, "awards")
    assert hash_vector_search(chunks, "awards")


def test_hybrid_search_uses_semantic_provider(monkeypatch):
    chunks = chunk_text("Awards and prizes. Membership requires achievement.", "u", "t")
    monkeypatch.setattr("eb1a.vector.semantic_search", lambda *args, **kwargs: hash_vector_search(chunks, "awards"))
    assert hybrid_search(chunks, "awards")
