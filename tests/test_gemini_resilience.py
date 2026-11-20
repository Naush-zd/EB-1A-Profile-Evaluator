from eb1a.providers import _coerce_structured_payload
from eb1a.schemas import BatchAssessmentPayload
from eb1a.criteria import CriterionId


def test_batch_parser_accepts_criteria_array_from_provider():
    payload = [
        {
            "criterion_id": criterion.value,
            "rating": "insufficient_info",
            "evidence": [],
            "reasoning": "No evidence.",
            "citations": [],
            "next_steps": ["Provide evidence."],
        }
        for criterion in CriterionId
    ]

    normalized = _coerce_structured_payload(payload, BatchAssessmentPayload)

    assert len(normalized["criteria"]) == 10
    assert normalized["criteria"][0]["criterion_id"] == CriterionId.AWARDS.value
