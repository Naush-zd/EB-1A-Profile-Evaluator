from dataclasses import dataclass
from .schemas import RetrievalResult, SourceChunk
from .criteria import CRITERIA
from .retrieve import bm25_search
from .vector import indexed_semantic_search, indexed_hybrid_search


@dataclass(frozen=True)
class RetrievalMetrics:
    queries: int
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float


def retrieval_metrics(results: list[list[RetrievalResult]], gold_chunk_ids: list[str]) -> RetrievalMetrics:
    if len(results) != len(gold_chunk_ids):
        raise ValueError("results and gold labels must have the same length")
    if not results:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0)
    hits3 = hits5 = mrr = 0.0
    for candidates, gold in zip(results, gold_chunk_ids):
        ids = [item.chunk.chunk_id for item in candidates]
        if gold in ids[:3]: hits3 += 1
        if gold in ids[:5]: hits5 += 1
        if gold in ids: mrr += 1 / (ids.index(gold) + 1)
    n = len(results)
    return RetrievalMetrics(n, hits3 / n, hits5 / n, mrr / n)


def criterion_gold_chunks(chunks: list[SourceChunk]) -> dict[str, str]:
    """Map each criterion to its official criterion-heading chunk."""
    mapping = {}
    for number, criterion in enumerate(CRITERIA, 1):
        candidates = [c for c in chunks if c.heading.lower().startswith(f"criterion {number}:")]
        if candidates:
            mapping[criterion.value] = candidates[0].chunk_id
    return mapping


def retrieval_benchmark(chunks: list[SourceChunk], index_path, embedding_model: str, reranker_model: str | None = None) -> dict:
    gold = criterion_gold_chunks(chunks)
    if len(gold) != len(CRITERIA):
        raise ValueError(f"retrieval gold mapping covers {len(gold)} of {len(CRITERIA)} criteria")
    queries = [(criterion.value, f"{criterion.value.replace('_', ' ')} {CRITERIA[criterion]}") for criterion in CRITERIA]
    lexical = [bm25_search(chunks, query, top_k=5) for _, query in queries]
    semantic = [indexed_semantic_search(chunks, query, index_path, top_k=5, model_name=embedding_model) for _, query in queries]
    hybrid = [indexed_hybrid_search(chunks, query, index_path, top_k=5, model_name=embedding_model) for _, query in queries]
    gold_ids = [gold[criterion_id] for criterion_id, _ in queries]
    report = {
        "gold_strategy": "first chunk whose heading starts with the numbered USCIS criterion heading",
        "lexical_bm25": retrieval_metrics(lexical, gold_ids).__dict__,
        "semantic_sentence_transformer": retrieval_metrics(semantic, gold_ids).__dict__,
        "hybrid_bm25_sentence_transformer": retrieval_metrics(hybrid, gold_ids).__dict__,
    }
    if reranker_model:
        from .vector import indexed_hybrid_search
        reranked = [indexed_hybrid_search(chunks, query, index_path, top_k=5, model_name=embedding_model, reranker_model=reranker_model) for _, query in queries]
        report["hybrid_with_cross_encoder"] = retrieval_metrics(reranked, gold_ids).__dict__
    return report
