from __future__ import annotations

import json
from typing import Iterable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings
from .llm import chat_model, embed_text
from .supabase_index import PageNode, index


def build_context_block(nodes: Sequence[PageNode], max_chars: int) -> str:
    lines: List[str] = []
    for idx, node in enumerate(nodes, start=1):
        snippet = (node.content or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        lines.append(
            f"[{idx}] {node.label} — page {node.page_number} (score {node.score:.3f})\n{snippet}"
        )
    return "\n\n".join(lines)


def citations_from_nodes(nodes: Sequence[PageNode]) -> List[dict]:
    citations: List[dict] = []
    for idx, node in enumerate(nodes, start=1):
        citations.append(
            {
                "id": idx,
                "label": node.label,
                "url": node.url,
                "page": node.page_number,
                "score": node.score,
            }
        )
    return citations


def retrieve(question: str, k: int) -> List[PageNode]:
    query_embedding = embed_text(question)
    return index.top_k(query_embedding, k=k)


def compose_messages(question: str, nodes: Sequence[PageNode]) -> List:
    settings = get_settings()
    context_block = build_context_block(nodes, settings.max_context_chars)
    sys = SystemMessage(
        content=(
            "You are a concise assistant answering questions about MMCD documents. "
            "Use ONLY the provided sources. If the answer is uncertain, say so briefly. "
            "Include inline citation markers like [1] that correspond to the numbered sources. "
            "Keep answers short (2-4 sentences)."
        )
    )
    user = HumanMessage(
        content=f"Question: {question}\n\nSources:\n{context_block}\n\nAnswer with citations."
    )
    return [sys, user]


def generate_answer(question: str, nodes: Sequence[PageNode]) -> str:
    messages = compose_messages(question, nodes)
    response = chat_model(streaming=False).invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def stream_answer(question: str, nodes: Sequence[PageNode]) -> Iterable[str]:
    messages = compose_messages(question, nodes)
    for chunk in chat_model(streaming=True).stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield str(chunk.content)


def generate_followups(question: str, answer: str, count: int = 3) -> List[str]:
    prompt = [
        SystemMessage(
            content=(
                "Propose concise follow-up questions (max 12 words each) that logically continue the conversation. "
                "Return ONLY a JSON array of strings."
            )
        ),
        HumanMessage(content=f"User asked: {question}\nAssistant answered: {answer}"),
    ]
    raw = chat_model(streaming=False).invoke(prompt)
    text = raw.content if hasattr(raw, "content") else str(raw)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data][:count]
    except Exception:
        pass
    # Fallback: split lines
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    return lines[:count]
