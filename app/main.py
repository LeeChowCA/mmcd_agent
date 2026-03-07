from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_settings
from .graph import AgentState, build_graph
from .rag import citations_from_nodes, generate_followups, retrieve, stream_answer
from .schemas import AgentResponse, ChatRequest, ChatMessage
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
