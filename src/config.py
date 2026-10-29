
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Enterprise Knowledge Intelligence System"
    app_env: str = "development"
    log_level: str = "INFO"

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval settings
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
