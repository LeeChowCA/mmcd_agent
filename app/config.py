from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI / model
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    chat_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    transcription_model: str = Field(default="gpt-4o-mini-transcribe")
    speech_model: str = Field(default="gpt-4o-mini-tts")
    speech_voice: str = Field(default="alloy")
    speech_format: str = Field(default="mp3")
    transcription_language: str = Field(default="en")
    transcription_prompt: str = Field(
        default=(
            "This audio is about MMCD, SGC, SGC pages, specifications, sewers, roadways, "
            "testing, commissioning, and civil construction terms."
        )
    )

    # Supabase
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_page_embeddings_table: str = Field(default="page_embeddings")
    supabase_document_pages_table: str = Field(default="document_pages")

    # Retrieval and responses
    top_k: int = 6
    max_context_chars: int = 1800
    recommended_questions: List[str] = Field(
        default_factory=lambda: [
            "Summarize the key updates in the latest MMCD handbook.",
            "What eligibility rules changed compared to last year?",
            "List the most important compliance deadlines I should know.",
        ]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
