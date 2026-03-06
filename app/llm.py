from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_settings


@lru_cache(maxsize=2)
def _chat_model(streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        temperature=0.2,
        streaming=streaming,
    )


def chat_model(streaming: bool = False) -> ChatOpenAI:
    # Cache is keyed only by streaming flag to keep two shared clients (streaming/non-streaming).
    return _chat_model(streaming=streaming)


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )


def embed_text(text: str) -> List[float]:
    return _embeddings().embed_query(text)
