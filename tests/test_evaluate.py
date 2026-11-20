from eb1a.evaluate import assess_profile, validate_grounding
from eb1a.profile import load_profile
from eb1a.corpus import chunk_text


def test_mock_assessment_is_structured_and_grounded():
    from eb1a.schemas import ProfileDocument
    profile = ProfileDocument(profile_id="p", source_path="x", content="Received a national award.\n", content_hash="h", pages=1)
    chunks = chunk_text("Awards are evidence of recognition by the field.", "uscis", "t")
    assessment = assess_profile(profile, chunks, "v1")
    assert len(assessment.criteria) == 10
    assert assessment.criteria[0].evidence
    assert validate_grounding(assessment, profile, chunks) == []
