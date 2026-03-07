from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langgraph.graph import END, StateGraph

from .config import get_settings
from .rag import citations_from_nodes, generate_answer, generate_followups, retrieve
from .supabase_index import PageNode


@dataclass
class AgentState:
    question: str
    contexts: List[PageNode] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    answer: str = ""
    followups: List[str] = field(default_factory=list)


def _retrieve_node(state: AgentState) -> AgentState:
    settings = get_settings()
    nodes = retrieve(state.question, k=settings.top_k)
    state.contexts = nodes
    state.citations = citations_from_nodes(nodes, state.question)
    return state


def _answer_node(state: AgentState) -> AgentState:
    state.answer = generate_answer(state.question, state.contexts)
    state.followups = generate_followups(state.question, state.answer)
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("answer", _answer_node)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.set_entry_point("retrieve")
    return graph.compile()
