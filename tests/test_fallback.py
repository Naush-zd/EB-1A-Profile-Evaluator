from eb1a.corpus import chunk_text
from eb1a.schemas import ProfileDocument
from eb1a.service import assess_with_fallback


class FailingProvider:
    provider_name = "groq"
    batch_mode = True
    network_call_count = 1

    def generate_structured(self, prompt, output_schema):
        raise RuntimeError("rate limit")


def test_provider_failure_returns_local_fallback():
    profile = ProfileDocument(profile_id="p", source_path="x", content="Received an award.", content_hash="h", pages=1)
    chunks = chunk_text("EB-1A extraordinary ability policy manual. Awards and prizes.", "u", "t")
    result = assess_with_fallback(profile, chunks, provider=FailingProvider(), skip_empty_evidence=True)
    assert result.run.fallback_used is True
    assert result.run.mode == "test_hash_baseline"
    assert len(result.criteria) == 10
