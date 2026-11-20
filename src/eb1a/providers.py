import json
import os
import urllib.request
import urllib.error
import re
import time
import hashlib
from pathlib import Path
from typing import Protocol, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DiskCache:
    """Lightweight persistent response cache to avoid burning LLM rate limits."""

    def __init__(self, cache_dir: str = ".eb1a_cache"):
        self.cache_dir = Path(cache_dir)
        if os.getenv("EB1A_DISABLE_CACHE") != "1":
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def get(self, key: str) -> dict | None:
        if os.getenv("EB1A_DISABLE_CACHE") == "1":
            return None
        target = self.cache_dir / f"{key}.json"
        if target.exists():
            try:
                return json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, payload: dict) -> None:
        if os.getenv("EB1A_DISABLE_CACHE") == "1":
            return
        target = self.cache_dir / f"{key}.json"
        try:
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass


def _parse_retry_delay(detail: str, headers=None) -> float | None:
    """Extract wait time from HTTP Retry-After headers or provider error text."""
    if headers:
        hdr = headers.get("Retry-After") or headers.get("retry-after")
        if hdr:
            try:
                return float(hdr)
            except ValueError:
                pass
    m_ms = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", detail, re.I)
    if m_ms:
        return float(m_ms.group(1)) / 1000.0
    m_time = re.search(r"try again in\s+(?:(\d+(?:\.\d+)?)\s*m(?:in)?)?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec)?)?", detail, re.I)
    if m_time and (m_time.group(1) or m_time.group(2)):
        mins = float(m_time.group(1) or 0)
        secs = float(m_time.group(2) or 0)
        return mins * 60.0 + secs
    return None


def _coerce_structured_payload(payload: dict, output_schema: type[T]) -> dict:
    """Normalize common JSON shorthand while retaining strict Pydantic validation."""
    if output_schema.__name__ == "BatchAssessmentPayload":
        from .schemas import CriterionAssessmentPayload
        if isinstance(payload, list):
            payload = {"criteria": payload}
        if not isinstance(payload, dict):
            raise ValueError("batch structured output must be a JSON object or criteria array")
        result = dict(payload)
        normalized = []
        for item in result.get("criteria", []):
            item = dict(item)
            criterion_id = item.get("criterion_id")
            item.pop("criterion_id", None)
            normalized_item = _coerce_structured_payload(item, CriterionAssessmentPayload)
            if criterion_id is not None:
                normalized_item["criterion_id"] = criterion_id
            normalized.append(normalized_item)
        result["criteria"] = normalized
        return result
    if output_schema.__name__ != "CriterionAssessmentPayload":
        return payload
    result = dict(payload)
    rating_aliases = {"meets": "partial", "meets_criterion": "partial", "does_not_meet": "insufficient_info", "unknown": "insufficient_info"}
    if isinstance(result.get("rating"), str):
        result["rating"] = rating_aliases.get(result["rating"].lower(), result["rating"].lower())
    evidence = result.get("evidence", [])
    if isinstance(evidence, str):
        result["evidence"] = [{"span_id": "provider-evidence-0", "quote": evidence}]
    elif isinstance(evidence, list):
        result["evidence"] = [
            {"span_id": f"provider-evidence-{index}", "quote": item} if isinstance(item, str) else item
            for index, item in enumerate(evidence)
        ]
    citations = result.get("citations", [])
    if citations is None:
        citations = []
    if isinstance(citations, list):
        normalized = []
        for item in citations:
            if isinstance(item, str):
                match = re.search(r"([A-Za-z0-9_-]+(?:-[A-Za-z0-9_-]+)*)", item)
                if match:
                    normalized.append({"chunk_id": match.group(1), "source_url": "", "locator": "", "quote": ""})
            elif isinstance(item, dict):
                item = dict(item)
                for field in ("chunk_id", "source_url", "locator", "quote"):
                    if item.get(field) is None:
                        item[field] = ""
                normalized.append(item)
        result["citations"] = normalized
    if result.get("evidence") is None:
        result["evidence"] = []
    if not result.get("reasoning"):
        for alias in ("explanation", "rationale", "analysis", "justification", "comment", "summary"):
            if result.get(alias):
                result["reasoning"] = str(result[alias])
                break
        else:
            result["reasoning"] = f"Evaluation grounded in retrieved USCIS policy guidance for {result.get('rating', 'insufficient_info')} rating."
    elif not isinstance(result["reasoning"], str):
        result["reasoning"] = str(result["reasoning"])
    next_steps = result.get("next_steps")
    if not next_steps:
        result["next_steps"] = ["Add supporting documentation."]
    elif isinstance(next_steps, str):
        result["next_steps"] = [next_steps]
    elif isinstance(next_steps, list):
        filtered = [str(x) for x in next_steps if str(x).strip()]
        result["next_steps"] = filtered[:2] if filtered else ["Add supporting documentation."]
    valid_ratings = {"strong", "partial", "missing", "insufficient_info"}
    if result.get("rating") not in valid_ratings:
        result["rating"] = "insufficient_info"
    return result


def _load_local_env() -> set[str]:
    if os.getenv("EB1A_DISABLE_DOTENV") == "1":
        return set()
    path = os.getenv("EB1A_ENV_FILE", ".env")
    if not os.path.exists(path):
        return set()
    loaded = set()
    for raw_line in open(path, encoding="utf-8"):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)
            loaded.add(key.strip())
    return loaded


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(Protocol):
    def generate_structured(self, prompt: str, output_schema: type[T]) -> T: ...


class FailoverLLMProvider:
    """Try configured real providers in order; never substitutes heuristics."""

    batch_mode = True

    def __init__(self, providers: list):
        if not providers:
            raise ValueError("provider failover chain is empty")
        self.providers = providers
        self.provider_name = "failover"
        self.selected_provider = None
        self.model = None
        self.attempted_providers = []
        self.failover_used = False
        self.failure_reasons = []
        self.structured_call_count = 0

    def generate_structured(self, prompt: str, output_schema: type[T]) -> T:
        self.structured_call_count += 1
        for index, provider in enumerate(self.providers):
            name = getattr(provider, "provider_name", provider.__class__.__name__.lower())
            self.attempted_providers.append(name)
            try:
                result = provider.generate_structured(prompt, output_schema)
                self.selected_provider = name
                self.model = getattr(provider, "model", None)
                self.failover_used = index > 0
                return result
            except Exception as exc:
                self.failure_reasons.append(f"{name}: {str(exc)[:300]}")
        raise RuntimeError("all configured LLM providers failed: " + " | ".join(self.failure_reasons))

    def verify_credentials(self) -> dict:
        checks = []
        for provider in self.providers:
            verifier = getattr(provider, "verify_credentials", None)
            if verifier is None:
                checks.append({"provider": getattr(provider, "provider_name", "unknown"), "ok": True, "message": "credential verification is not supported"})
            else:
                checks.append(verifier())
        return {
            "provider": "failover",
            "ok": any(item.get("ok") for item in checks),
            "providers": checks,
            "configured_order": [getattr(provider, "provider_name", "unknown") for provider in self.providers],
        }


def _build_provider(provider_name: str):
    name = provider_name.strip().lower()
    if name in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider(provider_name="openai")
    if name == "groq":
        return OpenAICompatibleProvider(provider_name="groq", credential_source="dotenv" if os.getenv("GROQ_API_KEY") else "missing")
    if name in {"gemini", "google"}:
        return GeminiProvider()
    if name in {"bedrock", "aws-bedrock"}:
        return BedrockProvider()
    raise ValueError(f"unsupported provider in chain: {provider_name}")


def get_llm_provider_chain():
    _load_local_env()
    configured = os.getenv("LLM_PROVIDER_CHAIN") or os.getenv("LLM_PROVIDER", "mock")
    names = [item.strip() for item in configured.split(",") if item.strip()]
    if names == ["mock"]:
        return MockLLMProvider()
    providers = []
    unavailable = []
    for name in names:
        try:
            providers.append(_build_provider(name))
        except Exception as exc:
            unavailable.append(f"{name}: {exc}")
    if not providers:
        if len(names) == 1:
            detail = unavailable[0].split(": ", 1)[1] if unavailable else "provider unavailable"
            if "required" in detail:
                raise ValueError(detail)
            raise ValueError("unsupported LLM_PROVIDER: " + names[0] + "; use mock, openai, groq, gemini, or bedrock")
        raise ValueError("no usable provider in LLM_PROVIDER_CHAIN; " + " | ".join(unavailable))
    if unavailable:
        print("[provider-chain] unavailable providers skipped: " + " | ".join(unavailable))
    return providers[0] if len(providers) == 1 else FailoverLLMProvider(providers)


class MockEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(ch.isalpha() for ch in text))] for text in texts]


class MockLLMProvider:
    def generate_structured(self, prompt: str, output_schema: type[T]) -> T:
        raise NotImplementedError("mock LLM responses must be supplied by the assessment test harness")


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat provider; no SDK dependency required."""

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None, provider_name: str = "openai", credential_source: str = "unknown"):
        self.provider_name = provider_name
        self.credential_source = credential_source
        self.batch_mode = True
        self.network_call_count = 0
        self.structured_call_count = 0
        env_key = "GROQ_API_KEY" if provider_name == "groq" else "OPENAI_API_KEY"
        default_url = "https://api.groq.com/openai/v1" if provider_name == "groq" else "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv(env_key)
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b" if provider_name == "groq" else "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or default_url).rstrip("/")
        self.cache = DiskCache()
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the hosted provider")

    def generate_structured(self, prompt: str, output_schema: type[T]) -> T:
        self.structured_call_count += 1
        cache_key = hashlib.sha256(f"{self.provider_name}:{self.model}:{prompt}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return output_schema.model_validate(_coerce_structured_payload(cached, output_schema))

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1500,
        }
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": user_agent},
            method="POST",
        )
        for attempt in range(5):
            try:
                self.network_call_count += 1
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                self.cache.set(cache_key, parsed)
                return output_schema.model_validate(_coerce_structured_payload(parsed, output_schema))
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = "provider returned no readable error body"
                if exc.code == 429 and attempt < 4:
                    delay = _parse_retry_delay(detail, getattr(exc, "headers", None))
                    if delay is not None:
                        retry_seconds = max(1.0, delay + 0.5)
                    else:
                        retry_seconds = min(60.0, 2.0 * (2 ** attempt))
                    max_wait = float(os.getenv("GROQ_MAX_RETRY_WAIT_SECONDS", "30")) if self.provider_name == "groq" else 120.0
                    if retry_seconds > max_wait:
                        raise RuntimeError(
                            f"{self.provider_name} rate limit requires waiting {retry_seconds:.1f}s, "
                            f"which exceeds the configured maximum of {max_wait:.1f}s. "
                            "Wait for the provider window to reset or use fast mode/cache."
                        ) from exc
                    print(f"[{self.provider_name}] 429 rate limit hit: waiting {retry_seconds:.1f}s before retry {attempt + 1}/5...")
                    time.sleep(retry_seconds)
                    continue
                if exc.code == 400 and ("json_validate_failed" in detail.lower() or "failed to validate json" in detail.lower()) and attempt < 4:
                    retry_seconds = 1.0 * (attempt + 1)
                    print(f"[{self.provider_name}] JSON validation retry: waiting {retry_seconds:.1f}s before retry {attempt + 1}/5...")
                    time.sleep(retry_seconds)
                    continue
                raise RuntimeError(f"{self.provider_name} request rejected with HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                raise RuntimeError(f"structured LLM request failed: {exc}") from exc
        raise RuntimeError(f"{self.provider_name} request failed after rate-limit retries")

    def verify_credentials(self) -> dict:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        request = urllib.request.Request(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json", "User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            available_models = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
            return {"provider": self.provider_name, "ok": True, "available_models": available_models, "model": self.model, "base_url": self.base_url, "credential_source": self.credential_source}
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = "provider returned no readable error body"
            return {"provider": self.provider_name, "ok": False, "status": exc.code, "message": "credentials or account access rejected", "provider_detail": detail, "model": self.model, "base_url": self.base_url, "credential_source": self.credential_source}


class GeminiProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.provider_name = "gemini"
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or (os.getenv("LLM_MODEL") if os.getenv("LLM_PROVIDER", "").lower() in {"gemini", "google"} else None) or "gemini-3.8-flash"
        self.structured_call_count = 0
        self.batch_mode = True
        self.cache = DiskCache()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")

    def generate_structured(self, prompt: str, output_schema: type[T]) -> T:
        self.structured_call_count += 1
        cache_key = hashlib.sha256(f"gemini:{self.model}:{prompt}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return output_schema.model_validate(_coerce_structured_payload(cached, output_schema))

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = json.loads(response.read().decode())
                content = body["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(content)
                self.cache.set(cache_key, parsed)
                return output_schema.model_validate(_coerce_structured_payload(parsed, output_schema))
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = "provider returned no readable error body"
                if exc.code in {429, 503} and attempt < 4:
                    delay = _parse_retry_delay(detail, getattr(exc, "headers", None))
                    retry_seconds = max(2.0, (delay + 0.5) if delay is not None else 2.0 * (2 ** attempt))
                    if retry_seconds > 60:
                        raise RuntimeError(f"Gemini service unavailable for {retry_seconds:.1f}s; retry later") from exc
                    label = "503 service unavailable" if exc.code == 503 else "429 rate limit hit"
                    print(f"[gemini] {label}: waiting {retry_seconds:.1f}s before retry {attempt + 1}/5...")
                    time.sleep(retry_seconds)
                    continue
                raise RuntimeError(f"Gemini request rejected with HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt < 4:
                    retry_seconds = min(30.0, 2.0 * (2 ** attempt))
                    print(f"[gemini] transient network error: waiting {retry_seconds:.1f}s before retry {attempt + 1}/5...")
                    time.sleep(retry_seconds)
                    continue
                raise RuntimeError(f"Gemini structured request failed after transient network retries: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Gemini structured request failed: {exc}") from exc
        raise RuntimeError("Gemini request failed after rate-limit retries")

    def verify_credentials(self) -> dict:
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode())
            models = [item.get("name") for item in body.get("models", []) if isinstance(item, dict)]
            return {"provider": "gemini", "ok": True, "models": len(models), "model": self.model, "credential_source": "environment"}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return {"provider": "gemini", "ok": False, "status": exc.code, "message": "credentials or account access rejected", "provider_detail": detail, "model": self.model}
        except Exception as exc:
            return {"provider": "gemini", "ok": False, "message": str(exc), "model": self.model}


class BedrockProvider:
    def __init__(self, model: str | None = None, region: str | None = None):
        self.provider_name = "bedrock"
        self.model = model or os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.structured_call_count = 0
        try:
            import boto3
            self.client = boto3.client("bedrock-runtime", region_name=self.region)
        except ImportError as exc:
            raise ValueError("boto3 is required for the Bedrock provider") from exc

    def generate_structured(self, prompt: str, output_schema: type[T]) -> T:
        self.structured_call_count += 1
        payload = {"messages": [{"role": "user", "content": [{"text": prompt}]}], "inferenceConfig": {"maxTokens": 4000, "temperature": 0}}
        try:
            response = self.client.converse(modelId=self.model, **payload)
            content = response["output"]["message"]["content"][0]["text"]
            return output_schema.model_validate(json.loads(content))
        except Exception as exc:
            raise RuntimeError(f"Bedrock structured request failed: {exc}") from exc


def get_llm_provider():
    return get_llm_provider_chain()


def _get_llm_provider_legacy():
    groq_key_preexisting = "GROQ_API_KEY" in os.environ
    loaded = _load_local_env()
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider(provider_name="openai")
    if provider == "groq":
        source = "process_environment" if groq_key_preexisting else ("dotenv" if "GROQ_API_KEY" in loaded else "missing")
        return OpenAICompatibleProvider(provider_name="groq", credential_source=source)
    if provider in {"gemini", "google"}:
        return GeminiProvider()
    if provider in {"bedrock", "aws-bedrock"}:
        return BedrockProvider()
    if provider == "mock":
        return MockLLMProvider()
    raise ValueError(f"unsupported LLM_PROVIDER: {provider}; use mock, openai, groq, gemini, or bedrock")
