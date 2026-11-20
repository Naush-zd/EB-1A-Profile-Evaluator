import os
import time
from datetime import datetime, timezone
import uuid
import json
from .prompts import PROMPT_VERSION
from .eval_metrics import classification_metrics
from pathlib import Path
from .evalset import compare_case, load_cases, summarize
from .profile import load_profile
from .corpus import read_jsonl
from .service import assess_with_fallback
from .providers import get_llm_provider, _load_local_env
from .metrics import retrieval_benchmark


def run_evaluation(dataset: str | Path, corpus: str | Path, fast: bool = False, index_path: str | Path | None = None, retrieval_mode: str = "hybrid") -> dict:
    chunks = read_jsonl(corpus)
    _load_local_env()
    provider = None if os.getenv("LLM_PROVIDER", "mock").lower() == "mock" else get_llm_provider()
    comparisons = []
    chunk_ids = {c.chunk_id for c in chunks}
    total_evidence_spans = 0
    hallucinated_spans = 0
    total_citations = 0
    valid_citations = 0
    quality_citations = 0
    total_llm_calls = 0
    fallback_count = 0

    cases = load_cases(dataset)
    for idx, case in enumerate(cases):
        if idx > 0 and provider is not None and getattr(provider, "provider_name", "") == "groq":
            case_delay = float(os.getenv("EB1A_CASE_INTERVAL_SECONDS", "1.5"))
            if case_delay > 0:
                time.sleep(case_delay)
        profile = load_profile(Path(dataset) / case["profile_file"])
        assessment = assess_with_fallback(profile, chunks, provider=provider, skip_empty_evidence=fast, allow_heuristic_fallback=provider is None, index_path=index_path, retrieval_mode="lexical" if provider is None else retrieval_mode)
        if assessment.run:
            total_llm_calls += assessment.run.llm_calls
            fallback_count += int(assessment.run.fallback_used)
        actual = {item.criterion_id.value: item.rating for item in assessment.criteria}
        comparisons.append(compare_case(case, actual))

        profile_text_lower = profile.content.lower()
        for item in assessment.criteria:
            for ev in item.evidence:
                total_evidence_spans += 1
                if ev.quote.lower() not in profile_text_lower:
                    hallucinated_spans += 1
            for cit in item.citations:
                total_citations += 1
                matching = next((chunk for chunk in chunks if chunk.chunk_id == cit.chunk_id), None)
                if matching is not None:
                    valid_citations += 1
                    if (cit.source_url == matching.source_url and cit.locator == matching.locator
                            and cit.quote.strip() and cit.quote.strip().lower() in matching.text.lower()):
                        quality_citations += 1

    summary = summarize(comparisons)
    summary["case_results"] = comparisons
    all_rows = [row for item in comparisons for row in item["results"]]
    summary["classification"] = classification_metrics(all_rows)
    summary["citation_validity_rate"] = round(valid_citations / total_citations, 3) if total_citations else 1.0
    summary["citation_quality_rate"] = round(quality_citations / total_citations, 3) if total_citations else 1.0
    summary["hallucinated_evidence_rate"] = round(hallucinated_spans / total_evidence_spans, 3) if total_evidence_spans else 0.0
    summary["llm_calls"] = total_llm_calls
    summary["fallback_count"] = fallback_count
    index_version = None
    if index_path and Path(index_path).exists():
        try:
            index_version = json.loads(Path(index_path).read_text(encoding="utf-8")).get("index_version")
        except (OSError, ValueError):
            index_version = None
    summary["run_metadata"] = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_version": chunks[0].corpus_version if chunks else "empty",
        "provider": getattr(provider, "provider_name", "mock") if provider is not None else "mock",
        "model": getattr(provider, "model", None) if provider is not None else None,
        "prompt_version": PROMPT_VERSION,
        "fast_mode": fast,
        "dataset_version": uuid.uuid5(uuid.NAMESPACE_URL, str(Path(dataset).resolve())).hex[:16],
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") if retrieval_mode != "lexical" else None,
        "retrieval_mode": retrieval_mode,
        "index_version": index_version,
        "index_path": str(Path(index_path).resolve()) if index_path else None,
        "authoritative": (fallback_count == 0 and provider is not None
                          and getattr(provider, "provider_name", "mock") != "mock"),
    }
    if index_path and Path(index_path).exists() and retrieval_mode != "lexical":
        summary["retrieval_benchmark"] = retrieval_benchmark(chunks, index_path, os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2")
    else:
        summary["retrieval_benchmark"] = {"status": "not_run", "reason": "semantic index not supplied"}
    return summary
