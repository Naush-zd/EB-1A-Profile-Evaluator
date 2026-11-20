from eb1a.criteria import CRITERION_QUERY_TERMS, CriterionId
from eb1a.corpus import chunk_text
from eb1a.metrics import retrieval_metrics
from eb1a.retrieve import bm25_search


def test_criterion_queries_cover_all_ten_criteria():
    assert set(CRITERION_QUERY_TERMS) == set(CriterionId)


def test_bm25_and_metrics():
    chunks = chunk_text("Awards prize recognition. Membership association achievement.", "u", "t")
    results = bm25_search(chunks, "award prize", top_k=5)
    metrics = retrieval_metrics([results], [chunks[0].chunk_id])
    assert metrics.recall_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
