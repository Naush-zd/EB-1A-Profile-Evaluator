from .criteria import CRITERIA, CriterionId
from .prompts import SYSTEM_INSTRUCTIONS
from .retrieve import search
from .schemas import EvidenceSpan, RetrievalResult


def build_batch_prompt(profile_text: str, evidence_by_criterion: dict, chunks: list, active_criteria=None) -> str:
    active_criteria = active_criteria or list(CRITERIA)
    blocks = []
    for criterion in active_criteria:
        definition = CRITERIA[criterion]
        evidence = evidence_by_criterion[criterion]
        retrieved = search(chunks, f"{criterion.value.replace('_', ' ')} {definition}", top_k=2)
        evidence_text = "\n".join(f"[{item.span_id}] {item.quote}" for item in evidence) or "NONE"
        source_text = "\n".join(f"[{item.chunk.chunk_id}] {item.chunk.text[:1800]}" for item in retrieved) or "NONE"
        blocks.append(f"Criterion ID: {criterion.value}\nDefinition: {definition}\nCandidate evidence:\n{evidence_text}\nUSCIS excerpts:\n{source_text}")
    return f"""{SYSTEM_INSTRUCTIONS}
Evaluate all ten criteria in one response.
Candidate profile:
{profile_text[:20000]}

{chr(10).join(blocks)}

Return one JSON object with a criteria array containing exactly one object for every criterion_id.
Each object must contain criterion_id, rating, evidence, reasoning, citations, and next_steps.
Ratings must be exactly strong, partial, missing, or insufficient_info.
Use partial for one unverified resume assertion; use strong only for corroborated or independently verifiable evidence.
Use insufficient_info when evidence is absent or uncertain. Use missing only when the profile affirmatively demonstrates the criterion is not met.
Evidence quotes must be exact profile text. Citation chunk_id values must be copied exactly from the USCIS excerpts."""
