import pytest

from eb1a.providers import FailoverLLMProvider
from eb1a.schemas import CriterionAssessmentPayload


class FailingProvider:
    provider_name = "first"
    model = "first-model"

    def generate_structured(self, prompt, output_schema):
        raise RuntimeError("temporary outage")


class WorkingProvider:
    provider_name = "second"
    model = "second-model"

    def generate_structured(self, prompt, output_schema):
        return output_schema(
            rating="insufficient_info",
            evidence=[],
            reasoning="No evidence.",
            citations=[],
            next_steps=["Provide evidence."],
        )


def test_failover_uses_next_provider_and_records_attempts():
    provider = FailoverLLMProvider([FailingProvider(), WorkingProvider()])

    result = provider.generate_structured("prompt", CriterionAssessmentPayload)

    assert result.rating == "insufficient_info"
    assert provider.selected_provider == "second"
    assert provider.attempted_providers == ["first", "second"]
    assert provider.failover_used is True


def test_failover_fails_explicitly_when_all_providers_fail():
    with pytest.raises(RuntimeError, match="all configured LLM providers failed"):
        FailoverLLMProvider([FailingProvider(), FailingProvider()]).generate_structured("prompt", CriterionAssessmentPayload)
