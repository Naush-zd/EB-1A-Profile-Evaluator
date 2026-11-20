from eb1a.corpus import chunk_text
from eb1a.evaluate import assess_profile
from eb1a.schemas import BatchAssessmentPayload, ProfileDocument
from eb1a.criteria import CriterionId


class BatchProvider:
    batch_mode = True

    def generate_structured(self, prompt, output_schema):
        assert output_schema is BatchAssessmentPayload
        return BatchAssessmentPayload(criteria=[{
            "criterion_id": criterion.value,
            "rating": "partial",
            "evidence": [],
            "reasoning": "Batch result.",
            "citations": [],
            "next_steps": ["Add evidence."],
        } for criterion in CriterionId])


def test_batch_provider_is_one_orchestrated_assessment():
    profile = ProfileDocument(profile_id="p", source_path="x", content="Received an award.", content_hash="h", pages=1)
    chunks = chunk_text("EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts.", "u", "t")
    result = assess_profile(profile, chunks, "v1", provider=BatchProvider(), skip_empty_evidence=True)
    assert len(result.criteria) == 10
