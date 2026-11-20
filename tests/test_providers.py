import os
import pytest
from eb1a.providers import MockLLMProvider, get_llm_provider


def test_mock_is_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("EB1A_DISABLE_DOTENV", "1")
    assert isinstance(get_llm_provider(), MockLLMProvider)


def test_groq_selects_openai_compatible_adapter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    provider = get_llm_provider()
    assert provider.provider_name == "groq"
    assert "groq.com" in provider.base_url


def test_missing_gemini_key_is_clear(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EB1A_DISABLE_DOTENV", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_llm_provider()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="mock, openai, groq, gemini, or bedrock"):
        get_llm_provider()


def test_parse_retry_delay():
    from eb1a.providers import _parse_retry_delay
    assert _parse_retry_delay("Please try again in 500ms") == 0.5
    assert _parse_retry_delay("Please try again in 4.5s") == 4.5
    assert _parse_retry_delay("Please try again in 1m20s") == 80.0
    assert _parse_retry_delay("Rate limit error", headers={"Retry-After": "6"}) == 6.0


def test_disk_cache(monkeypatch):
    from eb1a.providers import DiskCache
    from pathlib import Path
    import shutil
    import uuid
    cache_dir = Path(".test-tmp") / uuid.uuid4().hex / "cache"
    try:
        cache = DiskCache(cache_dir=str(cache_dir))
        cache.set("key1", {"val": 123})
        assert cache.get("key1") == {"val": 123}
        assert cache.get("nonexistent") is None
    finally:
        shutil.rmtree(cache_dir.parent, ignore_errors=True)
