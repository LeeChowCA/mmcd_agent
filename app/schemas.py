from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)


class VoiceTranscriptionResponse(BaseModel):
    text: str


class VoiceSpeechRequest(BaseModel):
    text: str
    voice: str | None = None


class Citation(BaseModel):
    id: int
    url: Optional[str] = None
    label: Optional[str] = None
    page: Optional[int] = None
    source_id: Optional[str] = None
    page_id: Optional[str] = None
    source_file: Optional[str] = None
    excerpt: Optional[str] = None
    matched_text: Optional[str] = None
    score: Optional[float] = None


class AgentResponse(BaseModel):
    answer: str
    citations: List[Citation]
    follow_up_questions: List[str]
    recommended_questions: List[str] | None = None
    messages: List[dict]
