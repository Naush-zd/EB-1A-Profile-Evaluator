from .criteria import CRITERIA, CriterionId
from .schemas import Assessment, AssessmentRun, BatchAssessmentPayload, CriterionAssessment, CriterionAssessmentPayload, EvidenceSpan, OverallAssessment, ProfileDocument, SourceCitation, RetrievalMetadata
from .retrieve import search
from .vector import indexed_hybrid_search, indexed_semantic_search
from .schemas import SourceChunk
from .prompts import build_criterion_prompt
from .batch import build_batch_prompt
import os
import time


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


def _profile_evidence(profile: ProfileDocument, criterion: CriterionId) -> list[EvidenceSpan]:
    terms = KEYWORDS[criterion]
    spans = []
    for index, line in enumerate(profile.content.splitlines()):
        lowered = line.lower()
        has_negative_context = any(marker in lowered for marker in ("no documented", "no evidence", "without evidence", "not documented", "none provided", "does not have"))
        if any(term in lowered for term in terms) and not has_negative_context:
            spans.append(EvidenceSpan(span_id=f"{profile.profile_id}-line-{index}", quote=line.strip()))
    return spans


def assess_profile(profile: ProfileDocument, chunks: list[SourceChunk], corpus_version: str = "unknown", provider=None, skip_empty_evidence: bool | None = None, retrieval_mode: str | None = None, index_path=None) -> Assessment:
    if skip_empty_evidence is None:
        skip_empty_evidence = os.getenv("EB1A_SKIP_EMPTY_EVIDENCE", "0") == "1"
    assessments = []
    retrieval_mode = retrieval_mode or "lexical"
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    def retrieve(query):
        if retrieval_mode == "hybrid":
            if not index_path:
                raise RuntimeError("production hybrid retrieval requires a persisted embedding index")
            return indexed_hybrid_search(chunks, query, index_path, top_k=3, model_name=embedding_model)
        if retrieval_mode == "semantic":
            if not index_path:
                raise RuntimeError("production semantic retrieval requires a persisted embedding index")
            return indexed_semantic_search(chunks, query, index_path, top_k=3, model_name=embedding_model)
        return search(chunks, query, top_k=3)
    if provider is not None and getattr(provider, "batch_mode", False):
        return assess_profile_batch(profile, chunks, corpus_version, provider, skip_empty_evidence, retrieval_mode, index_path)
    for index, (criterion, description) in enumerate(CRITERIA.items()):
        evidence = _profile_evidence(profile, criterion)
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
                    retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=3, embedding_model=embedding_model if retrieval_mode != "lexical" else None, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved])
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
            item = CriterionAssessment(criterion_id=criterion, retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=3, embedding_model=embedding_model if retrieval_mode != "lexical" else None, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved]), **payload_data)
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
    ranked = sorted(assessments, key=lambda item: {"strong": 3, "partial": 2, "missing": 1, "insufficient_info": 0}[item.rating], reverse=True)
    qualifying = sum(item.rating in {"strong", "partial"} for item in assessments)
    score = round(sum({"strong": 12, "partial": 7, "missing": 2, "insufficient_info": 0}[item.rating] for item in assessments) / 120 * 100, 1)
    label = "strong" if score >= 75 else "promising" if score >= 50 else "developing" if score >= 25 else "early"
    return Assessment(
        profile_id=profile.profile_id, corpus_version=corpus_version, criteria=assessments,
        overall=OverallAssessment(
            strongest_criteria=[x.criterion_id for x in ranked[:3] if x.rating in {"strong", "partial"}],
            main_gaps=[x.criterion_id.value for x in assessments if x.rating == "insufficient_info"],
            criteria_meeting_initial_threshold=qualifying, readiness_score=score, readiness_label=label,
            disclaimer="This is a prototype evidence-screening heuristic, not legal advice or a prediction of a USCIS outcome.",
        ),
    )


def assess_profile_batch(profile, chunks, corpus_version, provider, skip_empty_evidence=False, retrieval_mode="lexical", index_path=None):
    evidence_by = {criterion: _profile_evidence(profile, criterion) for criterion in CRITERIA}
    active_criteria = [criterion for criterion, evidence in evidence_by.items() if evidence] if skip_empty_evidence else list(CRITERIA)
    if not active_criteria:
        return assess_profile(profile, chunks, corpus_version, provider=None, skip_empty_evidence=True)
    payload = provider.generate_structured(build_batch_prompt(profile.content, evidence_by, chunks, active_criteria), BatchAssessmentPayload)
    result_by_id = {item.criterion_id: item for item in payload.criteria}
    assessments = []
    for criterion in CRITERIA:
        evidence = evidence_by[criterion]
        if criterion not in active_criteria:
            assessments.append(CriterionAssessment(criterion_id=criterion, rating="insufficient_info", evidence=[], reasoning=f"No explicit supporting evidence for {criterion.value.replace('_', ' ')} was found in the candidate profile.", citations=[], next_steps=["Add dated, independently verifiable supporting documentation."]))
            continue
        query = f"{criterion.value.replace('_', ' ')} {CRITERIA[criterion]}"
        if retrieval_mode == "hybrid":
            if not index_path:
                raise RuntimeError("production hybrid retrieval requires a persisted embedding index")
            retrieved = indexed_hybrid_search(chunks, query, index_path, top_k=2, model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        else:
            retrieved = search(chunks, query, top_k=2)
        raw = result_by_id.get(criterion)
        data = raw.model_dump() if raw else {"rating": "insufficient_info", "evidence": [], "reasoning": "No valid provider result was returned.", "citations": [], "next_steps": ["Add supporting documentation."]}
        data.pop("criterion_id", None)
        allowed_quotes = {item.quote for item in evidence}
        data["evidence"] = [item for item in data.get("evidence", []) if item.get("quote") in allowed_quotes]
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
        assessments.append(CriterionAssessment(criterion_id=criterion, retrieval=RetrievalMetadata(mode=retrieval_mode, top_k=2, embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") if retrieval_mode != "lexical" else None, retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved]), **data))
    ranked = sorted(assessments, key=lambda item: {"strong": 3, "partial": 2, "missing": 1, "insufficient_info": 0}[item.rating], reverse=True)
    qualifying = sum(item.rating in {"strong", "partial"} for item in assessments)
    score = round(sum({"strong": 12, "partial": 7, "missing": 2, "insufficient_info": 0}[item.rating] for item in assessments) / 120 * 100, 1)
    label = "strong" if score >= 75 else "promising" if score >= 50 else "developing" if score >= 25 else "early"
    return Assessment(profile_id=profile.profile_id, corpus_version=corpus_version, criteria=assessments, overall=OverallAssessment(strongest_criteria=[item.criterion_id for item in ranked[:3] if item.rating in {"strong", "partial"}], main_gaps=[item.criterion_id.value for item in assessments if item.rating == "insufficient_info"], criteria_meeting_initial_threshold=qualifying, readiness_score=score, readiness_label=label, disclaimer="This is a prototype evidence-screening heuristic, not legal advice or a prediction of a USCIS outcome."))


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
