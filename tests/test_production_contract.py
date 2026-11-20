import pytest
from eb1a.corpus import chunk_text
from eb1a.embedding_index import build_embedding_index, read_embedding_index
from eb1a.schemas import ProfileDocument
from eb1a.service import assess_with_fallback


class FakeEmbeddingProvider:
    model_name = "fake-test-model"
    dimension = 2

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_embedding_index_round_trip_and_stale_rejection():
    chunks = chunk_text("USCIS awards and prizes guidance.", "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2", "t")
    path = __import__("pathlib").Path(".test-embeddings.json")
    try:
        build_embedding_index(chunks, path, FakeEmbeddingProvider())
        payload = read_embedding_index(path, chunks, "fake-test-model")
        assert payload["dimension"] == 2
        changed = chunk_text("Changed USCIS awards guidance.", "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2", "t")
        with pytest.raises(ValueError, match="stale"):
            read_embedding_index(path, changed, "fake-test-model")
    finally:
        path.unlink(missing_ok=True)


def test_production_provider_failure_does_not_return_heuristic_result():
    class FailingProvider:
        provider_name = "groq"
        model = "test"
        def generate_structured(self, prompt, schema):
            raise RuntimeError("provider unavailable")

    profile = ProfileDocument(profile_id="p", source_path="x", content="Received an award.", content_hash="h", pages=1)
    chunks = chunk_text("EB-1A extraordinary ability policy manual. Awards and prizes.", "u", "t")
    with pytest.raises(RuntimeError, match="production assessment unavailable"):
        assess_with_fallback(profile, chunks, provider=FailingProvider(), allow_heuristic_fallback=False)
