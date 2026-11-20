from eb1a.evaluate import assess_profile, validate_grounding
from eb1a.corpus import chunk_text
from eb1a.providers import MockLLMProvider
from eb1a.schemas import CriterionAssessmentPayload, ProfileDocument


class DeterministicProvider:
    def generate_structured(self, prompt, output_schema):
        assert output_schema is CriterionAssessmentPayload
        return CriterionAssessmentPayload(
            rating="partial", evidence=[], reasoning="Provider response.", citations=[],
            next_steps=["Add supporting documentation."]
        )


def test_provider_path_runs_one_structured_assessment_per_criterion():
    profile = ProfileDocument(profile_id="p", source_path="x", content="A candidate profile.", content_hash="h", pages=1)
    chunks = chunk_text("EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts.", "u", "t")
    assessment = assess_profile(profile, chunks, "v1", provider=DeterministicProvider())
    assert len(assessment.criteria) == 10
    assert all(item.rating == "insufficient_info" for item in assessment.criteria)
    assert validate_grounding(assessment, profile, chunks) == []
