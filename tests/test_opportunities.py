from eb1a.criteria import CriterionId
from eb1a.opportunities import OpportunitySearchProvider, suggest_opportunities
from eb1a.schemas import Assessment


def valid_assessment():
    return {
        "profile_id": "p1",
        "corpus_version": "v1",
        "criteria": [
            {"criterion_id": criterion.value, "rating": "insufficient_info", "reasoning": "Not found.", "next_steps": ["Provide supporting evidence."]}
            for criterion in CriterionId
        ],
        "overall": {
            "strongest_criteria": [],
            "main_gaps": ["Evidence is incomplete"],
            "criteria_meeting_initial_threshold": 0,
            "readiness_score": 0,
            "readiness_label": "early",
            "disclaimer": "Prototype only.",
        },
    }


class FakeSearchProvider(OpportunitySearchProvider):
    provider_name = "fake"

    def __init__(self):
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        return [{"title": "Review panel call", "url": "https://example.org/call", "content": "Applications close 2030-01-01."}]


def test_opportunity_agent_uses_weak_criteria_not_profile_text():
    assessment = Assessment.model_validate(valid_assessment())
    provider = FakeSearchProvider()
    result = suggest_opportunities(assessment, "machine learning", provider)

    assert result["leads"][0]["status"] == "lead_only"
    assert "machine learning" in provider.queries[0]
    assert "candidate" not in provider.queries[0].lower()
    assert "resume" not in provider.queries[0].lower()
    assert result["disclaimer"].startswith("These are public opportunity leads")
