from __future__ import annotations

from typing import Tuple

import httpx

from .config import get_settings


def _api_base_url() -> str:
    settings = get_settings()
    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _auth_headers() -> dict[str, str]:
    settings = get_settings()
    return {"Authorization": f"Bearer {settings.openai_api_key}"}


async def transcribe_audio(
    audio_bytes: bytes,
    *,
    filename: str,
    content_type: str | None,
) -> str:
    settings = get_settings()
    data = {
        "model": settings.transcription_model,
        "response_format": "text",
        "language": settings.transcription_language,
    }
    if settings.transcription_prompt:
        data["prompt"] = settings.transcription_prompt

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"{_api_base_url()}/audio/transcriptions",
            headers=_auth_headers(),
            data=data,
            files={
                "file": (
                    filename or "voice-input.webm",
                    audio_bytes,
                    content_type or "application/octet-stream",
                )
            },
        )
        response.raise_for_status()

    transcript = response.text.strip()
    if not transcript:
        raise RuntimeError("Transcription returned an empty result.")
    return transcript


async def synthesize_speech(
    text: str,
    *,
    voice: str | None = None,
) -> Tuple[bytes, str]:
    settings = get_settings()
    payload = {
        "model": settings.speech_model,
        "voice": voice or settings.speech_voice,
        "input": text,
        "response_format": settings.speech_format,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{_api_base_url()}/audio/speech",
            headers={
                **_auth_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    media_type = response.headers.get("content-type") or "audio/mpeg"
    return response.content, media_type
