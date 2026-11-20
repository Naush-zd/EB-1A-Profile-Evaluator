from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .criteria import CriterionId

Rating = Literal["strong", "partial", "missing", "insufficient_info"]


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: str
    quote: str
    source: Literal["profile"] = "profile"


class ProfileDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str
    source_path: str
    content: str
    content_hash: str
    pages: int = Field(ge=1)
    suspicious_instructions: list[str] = Field(default_factory=list)
    page_text: list[str] = Field(default_factory=list)


class SourceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    text: str = Field(min_length=1)
    source_url: str
    retrieved_at: str
    corpus_version: str
    section: str
    heading: str = ""
    locator: str
    content_hash: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk: SourceChunk
    score: float
    matched_terms: list[str] = Field(default_factory=list)


class RetrievalMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["lexical", "semantic", "hybrid"]
    top_k: int = Field(ge=1)
    embedding_model: str | None = None
    index_version: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reranker_model: str | None = None


class SourceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    source_url: str
    locator: str
    quote: str


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: CriterionId
    rating: Rating
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reasoning: str
    citations: list[SourceCitation] = Field(default_factory=list)
    next_steps: list[str] = Field(min_length=1, max_length=2)
    retrieval: RetrievalMetadata | None = None


class CriterionAssessmentPayload(BaseModel):
    """Provider-facing payload; criterion identity is supplied by the orchestrator."""
    model_config = ConfigDict(extra="forbid")
    rating: Rating
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reasoning: str
    citations: list[SourceCitation] = Field(default_factory=list)
    next_steps: list[str] = Field(min_length=1, max_length=2)


class BatchCriterionAssessmentPayload(CriterionAssessmentPayload):
    criterion_id: CriterionId


class BatchAssessmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: list[BatchCriterionAssessmentPayload] = Field(min_length=1, max_length=10)


class AssessmentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str | None = None
    corpus_version: str
    prompt_version: str
    run_id: str
    timestamp: str
    mode: str
    llm_calls: int = Field(ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: int = Field(ge=0)
    embedding_model: str | None = None
    retrieval_mode: Literal["lexical", "semantic", "hybrid"] = "lexical"
    index_version: str | None = None


class OverallAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strongest_criteria: list[CriterionId]
    main_gaps: list[str]
    criteria_meeting_initial_threshold: int = Field(ge=0, le=10)
    readiness_score: float = Field(ge=0, le=100)
    readiness_label: Literal["early", "developing", "promising", "strong"]
    disclaimer: str


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "0.1"
    profile_id: str
    corpus_version: str
    criteria: list[CriterionAssessment] = Field(min_length=10, max_length=10)
    overall: OverallAssessment
    run: AssessmentRun | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        ids = [item.criterion_id for item in self.criteria]
        expected = set(CriterionId)
        if len(set(ids)) != 10 or set(ids) != expected:
            raise ValueError("criteria must contain each of the 10 criterion IDs exactly once")
        return self
