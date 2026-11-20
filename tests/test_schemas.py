import pytest
from pydantic import ValidationError
from eb1a.criteria import CriterionId
from eb1a.schemas import Assessment, CriterionAssessment


def criterion(cid):
    return {"criterion_id": cid, "rating": "insufficient_info", "reasoning": "Not found.", "next_steps": ["Provide supporting evidence."]}


def valid_assessment():
    return {"profile_id": "p1", "corpus_version": "v1", "criteria": [criterion(c.value) for c in CriterionId], "overall": {"strongest_criteria": [], "main_gaps": ["Evidence is incomplete"], "criteria_meeting_initial_threshold": 0, "readiness_score": 0, "readiness_label": "early", "disclaimer": "Prototype only."}}


def test_valid_assessment():
    assert Assessment.model_validate(valid_assessment()).profile_id == "p1"


def test_rejects_invalid_rating():
    data = valid_assessment(); data["criteria"][0]["rating"] = "certain"
    with pytest.raises(ValidationError): Assessment.model_validate(data)


def test_rejects_duplicate_criterion():
    data = valid_assessment(); data["criteria"] = data["criteria"][:-1] + [data["criteria"][0]]
    with pytest.raises(ValidationError): Assessment.model_validate(data)
