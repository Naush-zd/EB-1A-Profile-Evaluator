"""Candidate-profile evidence extraction.

Lexical matching is retained as a transparent baseline. Semantic matching uses
the configured Sentence Transformer and exact profile lines as the only quotes.
"""
import math
import os

from .criteria import CRITERIA, CRITERION_QUERY_TERMS, CriterionId
from .embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from .schemas import EvidenceSpan, ProfileDocument


NEGATIVE_CONTEXT = ("no documented", "no evidence", "without evidence", "not documented", "none provided", "does not have")


def lexical_profile_evidence(profile: ProfileDocument, criterion: CriterionId, keywords: dict[CriterionId, list[str]]) -> list[EvidenceSpan]:
    terms = keywords[criterion]
    spans = []
    for index, line in enumerate(profile.content.splitlines()):
        lowered = line.lower()
        if any(term in lowered for term in terms) and not any(marker in lowered for marker in NEGATIVE_CONTEXT):
            quote = line.strip()
            if quote:
                spans.append(EvidenceSpan(span_id=f"{profile.profile_id}-line-{index}", quote=quote))
    return spans


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def semantic_profile_evidence(
    profile: ProfileDocument,
    criterion: CriterionId,
    provider: EmbeddingProvider,
    threshold: float = 0.35,
    max_spans: int = 3,
) -> list[EvidenceSpan]:
    lines = [(index, line.strip()) for index, line in enumerate(profile.content.splitlines()) if len(line.strip()) >= 12]
    lines = [(index, line) for index, line in lines if not any(marker in line.lower() for marker in NEGATIVE_CONTEXT)]
    if not lines:
        return []
    query = f"{CRITERIA[criterion]} {CRITERION_QUERY_TERMS[criterion]}"
    vectors = provider.embed([query] + [line for _, line in lines])
    query_vector = vectors[0]
    scored = [(index, line, _cosine(query_vector, vector)) for (index, line), vector in zip(lines, vectors[1:])]
    selected = sorted((row for row in scored if row[2] >= threshold), key=lambda row: (-row[2], row[0]))[:max_spans]
    return [EvidenceSpan(span_id=f"{profile.profile_id}-line-{index}", quote=line) for index, line, _ in sorted(selected)]


def extract_profile_evidence(
    profile: ProfileDocument,
    criterion: CriterionId,
    keywords: dict[CriterionId, list[str]],
    mode: str = "lexical",
    provider: EmbeddingProvider | None = None,
) -> list[EvidenceSpan]:
    lexical = lexical_profile_evidence(profile, criterion, keywords)
    if mode == "lexical":
        return lexical
    if provider is None:
        provider = SentenceTransformerEmbeddingProvider(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    semantic = semantic_profile_evidence(
        profile,
        criterion,
        provider,
        threshold=float(os.getenv("EB1A_EVIDENCE_SIMILARITY_THRESHOLD", "0.35")),
    )
    if mode == "semantic":
        return semantic
    if mode == "hybrid":
        by_quote = {item.quote: item for item in lexical}
        by_quote.update({item.quote: item for item in semantic})
        return sorted(by_quote.values(), key=lambda item: int(item.span_id.rsplit("-", 1)[-1]))
    raise ValueError("evidence mode must be lexical, semantic, or hybrid")
