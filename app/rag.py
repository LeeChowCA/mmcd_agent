from __future__ import annotations

import json
import re
from typing import Iterable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings
from .llm import chat_model, embed_text
from .supabase_index import PageNode, index


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", value.lower())


def _clip_excerpt(text: str, start: int, length: int, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    left = max(0, start - half)
    right = min(len(text), start + length + half)

    if right - left < max_chars:
        if left == 0:
            right = min(len(text), max_chars)
        elif right == len(text):
            left = max(0, len(text) - max_chars)

    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{text[left:right].strip()}{suffix}"


def _expand_match_window(text: str, index: int, token_length: int, radius: int = 56) -> str:
    left = max(0, index - radius)
    right = min(len(text), index + token_length + radius)
    window = text[left:right].strip()
    return _normalize_space(window)


def _build_citation_snippet(question: str, content: str, max_chars: int = 260) -> tuple[str, str | None]:
    text = _normalize_space(content)
    if not text:
        return "", None

    normalized_question = _normalize_space(question)
    lower_text = text.lower()
    lower_question = normalized_question.lower()

    if normalized_question and lower_question in lower_text:
        index = lower_text.index(lower_question)
        matched_text = text[index : index + len(normalized_question)]
        return _clip_excerpt(text, index, len(normalized_question), max_chars), matched_text

    best_index = -1
    best_token = ""
    for token in _tokenize(question):
        if len(token) < 3:
            continue
        token_index = lower_text.find(token)
        if token_index == -1:
            continue
        if len(token) > len(best_token):
            best_index = token_index
            best_token = token

    if best_index >= 0 and best_token:
        matched_text = _expand_match_window(text, best_index, len(best_token))
        return _clip_excerpt(text, best_index, len(best_token), max_chars), matched_text

    fallback = text[:max_chars].strip()
    if len(text) > max_chars:
        fallback = f"{fallback}..."
    return fallback, None


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


def citations_from_nodes(nodes: Sequence[PageNode], question: str) -> List[dict]:
    citations: List[dict] = []
    for idx, node in enumerate(nodes, start=1):
        excerpt, matched_text = _build_citation_snippet(question, node.content)
        citations.append(
            {
                "id": idx,
                "label": node.label,
                "url": node.url,
                "page": node.page_number,
                "source_id": node.source_id,
                "page_id": node.page_id,
                "source_file": node.source_file,
                "excerpt": excerpt,
                "matched_text": matched_text,
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
