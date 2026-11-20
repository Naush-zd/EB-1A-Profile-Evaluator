from eb1a.batch import build_batch_prompt
from eb1a.criteria import CriterionId
from eb1a.evaluate import _normalize_provider_rating


def test_missing_without_grounded_evidence_becomes_insufficient_info():
    data = {
        "rating": "missing",
        "reasoning": "No evidence was found.",
        "citations": [],
        "next_steps": ["Provide evidence."],
    }

    normalized = _normalize_provider_rating(data, [])

    assert normalized["rating"] == "insufficient_info"
    assert "no exact, grounded profile evidence" in normalized["reasoning"]


def test_missing_with_grounded_evidence_is_preserved():
    evidence = [{"span_id": "p-line-1", "quote": "The candidate explicitly states they did not judge any work."}]
    data = {
        "rating": "missing",
        "reasoning": "The profile contains an affirmative contrary statement.",
        "citations": [],
        "next_steps": ["Provide evidence."],
    }

    normalized = _normalize_provider_rating(data, evidence)

    assert normalized["rating"] == "missing"
    assert normalized["evidence"] == evidence


def test_batch_prompt_contains_criterion_requirements_and_rating_procedure():
    prompt = build_batch_prompt(
        "A candidate profile.",
        {criterion: [] for criterion in CriterionId},
        [],
        active_criteria=[CriterionId.HIGH_SALARY],
        retrieved_by_criterion={CriterionId.HIGH_SALARY: []},
    )

    assert "reliable comparison showing it is high relative to others" in prompt
    assert "absence of documentation into missing" in prompt
    assert "missing only when the supplied profile contains grounded" in prompt
