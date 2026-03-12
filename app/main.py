from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_settings
from .graph import AgentState, build_graph
from .rag import citations_from_nodes, generate_followups, retrieve, stream_answer
from .schemas import (
    AgentResponse,
    ChatRequest,
    ChatMessage,
    VoiceSpeechRequest,
    VoiceTranscriptionResponse,
)
from .speech import synthesize_speech, transcribe_audio
from .supabase_index import index

settings = get_settings()
app = FastAPI(title="mmcd-agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


@app.get("/health")
def health():
    return {"ok": True, "service": "mmcd-agent"}


@app.get("/")
def root():
    return {
        "service": "mmcd-agent",
        "ok": True,
        "health": "/health",
        "docs": "/docs",
        "voice": {
            "transcribe": "/api/voice/transcribe",
            "speak": "/api/voice/speak",
        },
    }


def _last_user_question(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    return ""


@app.post("/api/ask", response_model=AgentResponse)
def ask(body: ChatRequest):
    question = _last_user_question(body.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user question provided.")

    state = graph.invoke(AgentState(question=question))
    reply_message = {"role": "assistant", "content": state.answer, "citations": state.citations}
    history = [msg.model_dump() for msg in body.messages] + [reply_message]

    recommended = settings.recommended_questions if len(body.messages) <= 1 else None

    return AgentResponse(
        answer=state.answer,
        citations=[c for c in state.citations],
        follow_up_questions=state.followups,
        recommended_questions=recommended,
        messages=history,
    )


@app.get('/graph/png')
def get_graph_png() -> Response:
    png_bytes = graph.get_graph().draw_mermaid_png()
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/ask/stream")
async def ask_stream(body: ChatRequest):
    question = _last_user_question(body.messages)
    if not question:
        return JSONResponse({"error": "No user question provided."}, status_code=400)

    nodes = retrieve(question, k=settings.top_k)
    citations = citations_from_nodes(nodes, question)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            yield (json.dumps({"event": "context", "citations": citations}) + "\n").encode("utf-8")
            assembled = ""
            for token in stream_answer(question, nodes):
                if not token:
                    continue
                assembled += token
                yield (json.dumps({"event": "token", "delta": token}) + "\n").encode("utf-8")

            followups = generate_followups(question, assembled)
            final = {
                "event": "done",
                "answer": assembled,
                "citations": citations,
                "follow_up_questions": followups,
            }
            yield (json.dumps(final) + "\n").encode("utf-8")
        except Exception as exc:  # pragma: no cover - best effort stream failure
            err = {"event": "error", "message": str(exc)}
            yield (json.dumps(err) + "\n").encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform"},
    )


@app.post("/api/voice/transcribe", response_model=VoiceTranscriptionResponse)
async def voice_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        transcript = await transcribe_audio(
            audio_bytes,
            filename=file.filename or "voice-input.webm",
            content_type=file.content_type,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Transcription request failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return VoiceTranscriptionResponse(text=transcript)


@app.post("/api/voice/speak")
async def voice_speak(body: VoiceSpeechRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required for speech synthesis.")

    try:
        audio_bytes, media_type = await synthesize_speech(text, voice=body.voice)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Speech synthesis failed: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="agent-reply.mp3"',
        },
    )
