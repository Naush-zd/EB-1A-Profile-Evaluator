import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    llm_model: str = "mock"
    embedding_model: str = "mock"
    uscis_source_url: str = "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2"
    data_dir: str = "data"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock"),
            llm_model=os.getenv("LLM_MODEL", "mock"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "mock"),
            uscis_source_url=os.getenv("USCIS_SOURCE_URL", cls.uscis_source_url),
            data_dir=os.getenv("EB1A_DATA_DIR", "data"),
        )
