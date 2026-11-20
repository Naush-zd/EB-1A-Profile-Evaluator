from .criteria import CRITERIA, CriterionId
from .schemas import Assessment, AssessmentRun, BatchAssessmentPayload, CriterionAssessment, CriterionAssessmentPayload, EvidenceSpan, OverallAssessment, ProfileDocument, SourceChunk, SourceCitation, RetrievalMetadata
from .retrieve import search
from .vector import indexed_hybrid_search, indexed_semantic_search
from .prompts import build_criterion_prompt
from .evidence import extract_profile_evidence
from .batch import build_batch_prompt
import os
import time

_RATING_WEIGHT = {"strong": 3, "partial": 2, "missing": 1, "insufficient_info": 0}
_RATING_SCORE = {"strong": 12, "partial": 7, "missing": 2, "insufficient_info": 0}


def _build_overall(profile_id: str, corpus_version: str, assessments: list[CriterionAssessment]) -> Assessment:
    ranked = sorted(assessments, key=lambda item: _RATING_WEIGHT[item.rating], reverse=True)
    qualifying = sum(item.rating in {"strong", "partial"} for item in assessments)
    score = round(sum(_RATING_SCORE[item.rating] for item in assessments) / 120 * 100, 1)
    label = "strong" if score >= 75 else "promising" if score >= 50 else "developing" if score >= 25 else "early"
    return Assessment(
        profile_id=profile_id,
        corpus_version=corpus_version,
        criteria=assessments,
        overall=OverallAssessment(
            strongest_criteria=[x.criterion_id for x in ranked[:3] if x.rating in {"strong", "partial"}],
            main_gaps=[x.criterion_id.value for x in assessments if x.rating == "insufficient_info"],
            criteria_meeting_initial_threshold=qualifying,
            readiness_score=score,
            readiness_label=label,
            disclaimer="This is a prototype evidence-screening heuristic, not legal advice or a prediction of a USCIS outcome.",
        ),
    )


def _normalize_provider_rating(data: dict, grounded_evidence: list[dict]) -> dict:
    """Fail closed when a provider rates a criterion without usable profile evidence."""
    data["evidence"] = grounded_evidence
    if not grounded_evidence and data.get("rating") != "insufficient_info":
        data["rating"] = "insufficient_info"
        data["reasoning"] = "The provider returned no exact, grounded profile evidence for this criterion; the result is therefore insufficient information rather than a finding that the criterion is missing."
    return data


KEYWORDS = {
    CriterionId.AWARDS: ["award", "prize", "honor"],
    CriterionId.MEMBERSHIP: ["membership", "member", "association"],
    CriterionId.PUBLISHED_MATERIAL: ["featured", "profiled", "interview", "published about"],
    CriterionId.JUDGING: ["judge", "judging", "reviewer", "reviewed submissions"],
    CriterionId.ORIGINAL_CONTRIBUTIONS: ["original", "patent", "adopted", "major significance"],
    CriterionId.SCHOLARLY_ARTICLES: ["paper", "publication", "journal", "scholarly", "authored"],
    CriterionId.EXHIBITIONS: ["exhibition", "showcase", "gallery", "displayed"],
    CriterionId.LEADING_CRITICAL_ROLE: ["led", "leading", "critical role", "director", "executive"],
    CriterionId.HIGH_SALARY: ["salary", "compensation", "remuneration", "paid"],
    CriterionId.COMMERCIAL_SUCCESS: ["box office", "sales", "revenue", "commercial success", "performing"],
}


def _profile_evidence(profile: ProfileDocument, criterion: CriterionId, evidence_mode: str = "lexical", evidence_provider=None) -> list[EvidenceSpan]:
    return extract_profile_evidence(profile, criterion, KEYWORDS, evidence_mode, evidence_provider)


def assess_profile(profile: ProfileDocument, chunks: list[SourceChunk], corpus_version: str = "unknown", provider=None, skip_empty_evidence: bool | None = None, retrieval_mode: str | None = None, index_path=None, evidence_mode: str | None = None) -> Assessment:
    if skip_empty_evidence is None:
        skip_empty_evidence = os.getenv("EB1A_SKIP_EMPTY_EVIDENCE", "0") == "1"
    assessments = []
    retrieval_mode = retrieval_mode or "lexical"
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.getenv("RERANKER_MODEL") or None
    evidence_mode = evidence_mode or os.getenv("EB1A_EVIDENCE_MODE", "lexical")
    evidence_provider = None
    if evidence_mode in {"semantic", "hybrid"}:
        from .embeddings import SentenceTransformerEmbeddingProvider
        evidence_provider = SentenceTransformerEmbeddingProvider(embedding_model)
    def retrieve(query):
        if retrieval_mode == "hybrid":
            if not index_path:
                raise RuntimeError("production hybrid retrieval requires a persisted embedding index")
            return indexed_hybrid_search(chunks, query, index_path, top_k=3, model_name=embedding_model, reranker_model=reranker_model)
        if retrieval_mode == "semantic":
            if not index_path:
                raise RuntimeError("production semantic retrieval requires a persisted embedding index")
            return indexed_semantic_search(chunks, query, index_path, top_k=3, model_name=embedding_model)
        return search(chunks, query, top_k=3)
    if provider is not None and getattr(provider, "batch_mode", False):
        return assess_profile_batch(profile, chunks, corpus_version, provider, skip_empty_evidence, retrieval_mode, index_path, evidence_mode, evidence_provider)
    for index, (criterion, description) in enumerate(CRITERIA.items()):
        evidence = _profile_evidence(profile, criterion, evidence_mode, evidence_provider)
        query = f"{criterion.value.replace('_', ' ')} {description}"
        retrieved = retrieve(query)
        citations = [SourceCitation(chunk_id=r.chunk.chunk_id, source_url=r.chunk.source_url, locator=r.chunk.locator, quote=r.chunk.text[:500]) for r in retrieved]
        if provider is not None:
            if skip_empty_evidence and not evidence:
                assessments.append(CriterionAssessment(
                    criterion_id=criterion,
                    rating="insufficient_info",
                    evidence=[],
                    reasoning=f"No explicit supporting evidence for {criterion.value.replace('_', ' ')} was found in the candidate profile.",
                    citations=citations[:1] if citations else [],
                    next_steps=["Add dated, independently verifiable supporting documentation."],
                    retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=3, embedding_model=embedding_model if retrieval_mode != "lexical" else None, reranker_model=reranker_model, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved])
                ))
                continue
            if index and getattr(provider, "provider_name", "") == "groq":
                interval = float(os.getenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "1.5"))
                if interval > 0:
                    time.sleep(interval)
            payload = provider.generate_structured(build_criterion_prompt(criterion, evidence, retrieved), CriterionAssessmentPayload)
            payload_data = payload.model_dump()
            chunk_by_id = {result.chunk.chunk_id: result.chunk for result in retrieved}
            enriched_citations = []
            for citation in payload_data.get("citations", []):
                chunk = chunk_by_id.get(citation["chunk_id"])
                if chunk:
                    citation.update(source_url=chunk.source_url, locator=chunk.locator, quote=chunk.text[:500])
                    enriched_citations.append(citation)
            if not enriched_citations and retrieved:
                top_chunk = retrieved[0].chunk
                enriched_citations.append({
                    "chunk_id": top_chunk.chunk_id,
                    "source_url": top_chunk.source_url,
                    "locator": top_chunk.locator,
                    "quote": top_chunk.text[:500],
                })
            payload_data["citations"] = enriched_citations
            grounded_evidence = [
                item for item in payload_data.get("evidence", [])
                if any(item.get("quote") == allowed.quote for allowed in evidence)
            ]
            payload_data = _normalize_provider_rating(payload_data, grounded_evidence)
            item = CriterionAssessment(criterion_id=criterion, retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=3, embedding_model=embedding_model if retrieval_mode != "lexical" else None, reranker_model=reranker_model, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved]), **payload_data)
            assessments.append(item)
            continue
        if evidence:
            rating = "partial" if len(evidence) == 1 else "strong"
            reasoning = "The profile contains explicit candidate text matching this criterion. Additional documentation is needed to establish the full legal requirement."
        else:
            rating = "insufficient_info"
            reasoning = "No explicit supporting evidence for this criterion was found in the supplied profile."
        assessments.append(CriterionAssessment(
            criterion_id=criterion, rating=rating, evidence=evidence, reasoning=reasoning,
            citations=citations, next_steps=["Add dated, independently verifiable supporting documentation."], retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=3, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved])
        ))
    return _build_overall(profile.profile_id, corpus_version, assessments)


def assess_profile_batch(profile, chunks, corpus_version, provider, skip_empty_evidence=False, retrieval_mode="lexical", index_path=None, evidence_mode="lexical", evidence_provider=None):
    evidence_by = {criterion: _profile_evidence(profile, criterion, evidence_mode, evidence_provider) for criterion in CRITERIA}
    active_criteria = [criterion for criterion, evidence in evidence_by.items() if evidence] if skip_empty_evidence else list(CRITERIA)
    if not active_criteria:
        return assess_profile(profile, chunks, corpus_version, provider=None, skip_empty_evidence=True)
    retrieved_by = {}
    for criterion in active_criteria:
        query = f"{criterion.value.replace('_', ' ')} {CRITERIA[criterion]}"
        if retrieval_mode == "hybrid":
            if not index_path:
                raise RuntimeError("production hybrid retrieval requires a persisted embedding index")
            retrieved_by[criterion] = indexed_hybrid_search(chunks, query, index_path, top_k=2, model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), reranker_model=os.getenv("RERANKER_MODEL") or None)
        else:
            retrieved_by[criterion] = search(chunks, query, top_k=2)
    payload = provider.generate_structured(build_batch_prompt(profile.content, evidence_by, chunks, active_criteria, retrieved_by), BatchAssessmentPayload)
    result_by_id = {item.criterion_id: item for item in payload.criteria}
    assessments = []
    for criterion in CRITERIA:
        evidence = evidence_by[criterion]
        if criterion not in active_criteria:
            assessments.append(CriterionAssessment(criterion_id=criterion, rating="insufficient_info", evidence=[], reasoning=f"No explicit supporting evidence for {criterion.value.replace('_', ' ')} was found in the candidate profile.", citations=[], next_steps=["Add dated, independently verifiable supporting documentation."]))
            continue
        retrieved = retrieved_by[criterion]
        raw = result_by_id.get(criterion)
        data = raw.model_dump() if raw else {"rating": "insufficient_info", "evidence": [], "reasoning": "No valid provider result was returned.", "citations": [], "next_steps": ["Add supporting documentation."]}
        data.pop("criterion_id", None)
        allowed_quotes = {item.quote for item in evidence}
        grounded_evidence = [item for item in data.get("evidence", []) if item.get("quote") in allowed_quotes]
        data = _normalize_provider_rating(data, grounded_evidence)
        chunk_by_id = {item.chunk.chunk_id: item.chunk for item in retrieved}
        citations = []
        for citation in data.get("citations", []):
            chunk = chunk_by_id.get(citation["chunk_id"])
            if chunk:
                citations.append({"chunk_id": chunk.chunk_id, "source_url": chunk.source_url, "locator": chunk.locator, "quote": chunk.text[:500]})
        if not citations and retrieved:
            chunk = retrieved[0].chunk
            citations.append({"chunk_id": chunk.chunk_id, "source_url": chunk.source_url, "locator": chunk.locator, "quote": chunk.text[:500]})
        data["citations"] = citations
        assessments.append(CriterionAssessment(criterion_id=criterion, retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=2, embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") if retrieval_mode != "lexical" else None, reranker_model=os.getenv("RERANKER_MODEL") or None, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved]), **data))
    return _build_overall(profile.profile_id, corpus_version, assessments)


def validate_grounding(assessment: Assessment, profile: ProfileDocument, chunks: list[SourceChunk]) -> list[str]:
    errors = []
    profile_text = profile.content.lower()
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    for item in assessment.criteria:
        for evidence in item.evidence:
            if evidence.quote.lower() not in profile_text:
                errors.append(f"{item.criterion_id}: evidence span is absent from profile")
        for citation in item.citations:
            chunk = chunk_map.get(citation.chunk_id)
            if not chunk:
                errors.append(f"{item.criterion_id}: invalid citation {citation.chunk_id}")
            elif not citation.source_url or not citation.locator or not citation.quote:
                errors.append(f"{item.criterion_id}: incomplete citation metadata {citation.chunk_id}")
            elif citation.quote.lower() not in chunk.text.lower():
                errors.append(f"{item.criterion_id}: citation quote is not present in chunk {citation.chunk_id}")
            elif item.retrieval and citation.chunk_id not in item.retrieval.retrieved_chunk_ids:
                errors.append(f"{item.criterion_id}: citation was not in retrieved results {citation.chunk_id}")
    return errors
