from eb1a.corpus import chunk_text
from eb1a.schemas import RetrievalResult
from eb1a.rerank import rerank


def test_rerank_orders_cross_encoder_scores(monkeypatch):
    chunks = chunk_text("Awards and prizes. Membership associations.", "u", "t")
    results = [RetrievalResult(chunk=chunk, score=0.1, matched_terms=[]) for chunk in chunks]

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return list(reversed(range(len(pairs))))

    monkeypatch.setattr("eb1a.rerank._load_reranker", lambda name: FakeModel())
    ranked = rerank("awards", results, "fake-reranker", 2)
    assert ranked[0].score >= ranked[-1].score
