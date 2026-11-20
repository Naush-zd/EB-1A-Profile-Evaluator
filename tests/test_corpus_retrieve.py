from eb1a.corpus import chunk_text
from eb1a.retrieve import search


def test_chunking_is_stable_for_fixed_timestamp():
    text = "Criterion 1 awards. The candidate received a national prize.\nCriterion 2 membership. The association requires outstanding achievement."
    a = chunk_text(text, "https://example.test", retrieved_at="2026-01-01T00:00:00+00:00", target_words=10)
    b = chunk_text(text, "https://example.test", retrieved_at="2026-01-01T00:00:00+00:00", target_words=10)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert a and all(x.content_hash for x in a)


def test_retrieval_is_ranked_and_deterministic():
    chunks = chunk_text("Awards prize national recognition.\n\nMembership association outstanding achievement.", "u", "t", target_words=20)
    results = search(chunks, "national prize award", top_k=2)
    assert results
    assert "prize" in results[0].matched_terms
    assert results == search(chunks, "national prize award", top_k=2)
