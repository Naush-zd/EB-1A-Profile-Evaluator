from eb1a.criteria import CriterionId
from eb1a.evaluate import KEYWORDS
from eb1a.evidence import extract_profile_evidence
from eb1a.schemas import ProfileDocument


class FakeEmbeddingProvider:
    model_name = "fake"
    dimension = 2

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "program committee" in lowered or "reviewed submissions" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        vectors[0] = [1.0, 0.0]
        return vectors


def profile(content: str) -> ProfileDocument:
    return ProfileDocument(profile_id="p", source_path="x", content=content, content_hash="h", pages=1)


def test_semantic_mode_finds_paraphrased_judging_evidence():
    result = extract_profile_evidence(
        profile("Served on the international program committee and reviewed submissions."),
        CriterionId.JUDGING,
        KEYWORDS,
        mode="semantic",
        provider=FakeEmbeddingProvider(),
    )

    assert [item.quote for item in result] == ["Served on the international program committee and reviewed submissions."]


def test_hybrid_mode_keeps_lexical_and_semantic_spans_without_duplicates():
    result = extract_profile_evidence(
        profile("Received an award for excellence.\nServed as a judge for a regional competition.\nServed on the international program committee."),
        CriterionId.JUDGING,
        KEYWORDS,
        mode="hybrid",
        provider=FakeEmbeddingProvider(),
    )

    assert [item.quote for item in result] == [
        "Served as a judge for a regional competition.",
        "Served on the international program committee.",
    ]
