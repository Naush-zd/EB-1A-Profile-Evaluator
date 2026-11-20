from eb1a.metrics import retrieval_benchmark


def test_retrieval_benchmark_signature_accepts_optional_reranker():
    assert "reranker_model" in retrieval_benchmark.__annotations__
