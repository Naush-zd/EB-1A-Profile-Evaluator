import time
import uuid
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from .evaluate import assess_profile
from .schemas import AssessmentRun
from .prompts import PROMPT_VERSION


def assess_with_fallback(profile, chunks, provider=None, skip_empty_evidence=True, allow_heuristic_fallback=True, index_path=None, retrieval_mode=None, evidence_mode=None):
    if provider is None and not allow_heuristic_fallback:
        raise RuntimeError("production assessment requires a configured LLM provider; set LLM_PROVIDER and credentials")
    started = time.perf_counter()
    provider_name = getattr(provider, "selected_provider", None) or getattr(provider, "provider_name", "mock") if provider is not None else "mock"
    before = getattr(provider, "structured_call_count", 0) if provider is not None else 0
    selected_mode = retrieval_mode or "lexical"
    selected_index_version = None
    if index_path and Path(index_path).exists():
        try:
            selected_index_version = json.loads(Path(index_path).read_text(encoding="utf-8")).get("index_version")
        except (OSError, ValueError):
            selected_index_version = None
    try:
        selected_evidence_mode = evidence_mode or os.getenv("EB1A_EVIDENCE_MODE", "lexical")
        assessment = assess_profile(profile, chunks, chunks[0].corpus_version if chunks else "empty", provider=provider, skip_empty_evidence=skip_empty_evidence, index_path=index_path, retrieval_mode=retrieval_mode, evidence_mode=selected_evidence_mode)
        calls = max(0, getattr(provider, "structured_call_count", 0) - before) if provider is not None else 0
        selected_provider_name = getattr(provider, "selected_provider", None) or provider_name
        run = AssessmentRun(provider=selected_provider_name, model=getattr(provider, "model", None), corpus_version=chunks[0].corpus_version if chunks else "empty", prompt_version=PROMPT_VERSION, run_id=uuid.uuid4().hex, timestamp=datetime.now(timezone.utc).isoformat(), mode="production_llm" if provider is not None else "test_mock", llm_calls=calls, latency_ms=int((time.perf_counter() - started) * 1000), embedding_model=(os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2") if selected_mode != "lexical" else None, retrieval_mode=selected_mode, evidence_mode=selected_evidence_mode, provider_chain=getattr(provider, "attempted_providers", []) if provider is not None else [], provider_failover_used=bool(getattr(provider, "failover_used", False)), index_version=selected_index_version)
        return assessment.model_copy(update={"run": run})
    except Exception as exc:
        if not allow_heuristic_fallback:
            raise RuntimeError(f"production assessment unavailable: {exc}") from exc
        fallback = assess_profile(profile, chunks, chunks[0].corpus_version if chunks else "empty", provider=None, skip_empty_evidence=True, evidence_mode=selected_evidence_mode)
        run = AssessmentRun(provider=provider_name, model=getattr(provider, "model", None), corpus_version=chunks[0].corpus_version if chunks else "empty", prompt_version=PROMPT_VERSION, run_id=uuid.uuid4().hex, timestamp=datetime.now(timezone.utc).isoformat(), mode="test_hash_baseline", llm_calls=max(0, getattr(provider, "structured_call_count", 0) - before) if provider is not None else 0, fallback_used=True, fallback_reason=str(exc)[:500], latency_ms=int((time.perf_counter() - started) * 1000), embedding_model=(os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2") if selected_evidence_mode != "lexical" else None, retrieval_mode="lexical", evidence_mode=selected_evidence_mode, index_version=selected_index_version)
        return fallback.model_copy(update={"run": run})
