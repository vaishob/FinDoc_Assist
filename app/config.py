from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DATA_DIR = Path("/tmp/findoc-assist") if os.getenv("VERCEL") else BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = Field(default="FinDoc Assist", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    database_path: Path = Field(default=RUNTIME_DATA_DIR / "findoc.db", alias="DATABASE_PATH")
    upload_dir: Path = Field(default=RUNTIME_DATA_DIR / "uploads", alias="UPLOAD_DIR")
    vector_index_path: Path = Field(default=RUNTIME_DATA_DIR / "vector_index", alias="VECTOR_INDEX_PATH")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_model: str = Field(default="llama-3.1-8b-instant", alias="LLM_MODEL")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    top_k: int = Field(default=10, alias="TOP_K")
    final_context_k: int = Field(default=5, alias="FINAL_CONTEXT_K")
    similarity_threshold: float = Field(default=0.25, alias="SIMILARITY_THRESHOLD")
    pii_mode: str = Field(default="mask", alias="PII_MODE")
    secret_key: str = Field(default="change-me", alias="SECRET_KEY")


settings = Settings()
