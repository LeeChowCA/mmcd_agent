from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import httpx
import numpy as np

from .config import get_settings


@dataclass
class PageNode:
    page_id: str
    page_number: int
    document_id: str | None
    content: str
    source_file: str
    source_id: str
    url: str
    label: str
    embedding: np.ndarray
    norm: float
    metadata: Dict[str, Any]
    score: float = 0.0


class SupabaseIndex:
    """
    Lightweight in-memory index built from Supabase page_embeddings + document_pages.
    Refreshes lazily with a short TTL to avoid repeated network calls.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._cached_at: float = 0.0
        self._nodes: List[PageNode] = []
        self._settings = get_settings()

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self._settings.supabase_service_role_key,
            "Authorization": f"Bearer {self._settings.supabase_service_role_key}",
        }

    def _fetch_page_embeddings(self) -> List[Dict[str, Any]]:
        url = f"{self._settings.supabase_url}/rest/v1/{self._settings.supabase_page_embeddings_table}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params={"select": "*"}, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def _fetch_document_pages(self) -> Dict[str, Dict[str, Any]]:
        url = f"{self._settings.supabase_url}/rest/v1/{self._settings.supabase_document_pages_table}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params={"select": "id,document_id,page_number,content,metadata"}, headers=self._headers())
            resp.raise_for_status()
            rows = resp.json()
            return {row["id"]: row for row in rows}

    @staticmethod
    def _parse_embedding(raw: Any) -> np.ndarray:
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, str):
            data = json.loads(raw)
        else:
            raise ValueError("Unsupported embedding format from Supabase.")
        return np.array(data, dtype=np.float32)

    @staticmethod
    def _to_label(source_file: str) -> str:
        base = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        base = base.rsplit(".", 1)[0] if "." in base else base
        if not base:
            return "Document"
        return base.replace("_", " ").replace("-", " ").strip().title()

    @staticmethod
    def _to_source_id(source_file: str) -> str:
        base = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        base = base.rsplit(".", 1)[0] if "." in base else base
        slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
        return slug or "document"

    def _hydrate(self) -> None:
        embeddings = self._fetch_page_embeddings()
        pages_by_id = self._fetch_document_pages()
        nodes: List[PageNode] = []

        for row in embeddings:
            page_id = row.get("page_id")
            if not page_id or page_id not in pages_by_id:
                continue

            page_row = pages_by_id[page_id]
            metadata = row.get("metadata") or {}
            embedding = self._parse_embedding(row.get("embedding"))
            norm = float(np.linalg.norm(embedding)) or 1.0

            page_number = int(metadata.get("page_number") or page_row.get("page_number") or 1)
            source_file = metadata.get("source_file") or metadata.get("file_name") or ""
            if not source_file:
                source_file = "document.pdf"

            source_file_leaf = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            url = metadata.get("file_url") or f"/{quote(source_file_leaf)}#page={page_number}"
            label = self._to_label(source_file)
            source_id = self._to_source_id(source_file)

            nodes.append(
                PageNode(
                    page_id=page_id,
                    document_id=str(page_row.get("document_id")) if page_row.get("document_id") else None,
                    page_number=page_number,
                    content=page_row.get("content") or "",
                    source_file=source_file,
                    source_id=source_id,
                    url=url,
                    label=label,
                    embedding=embedding,
                    norm=norm,
                    metadata={
                        "source_file": source_file,
                        "page_number": page_number,
                        **(page_row.get("metadata") or {}),
                        **(metadata if isinstance(metadata, dict) else {}),
                    },
                )
            )

        self._nodes = nodes
        self._cached_at = time.time()

    def _ensure_ready(self) -> None:
        if not self._nodes or (time.time() - self._cached_at) > self.ttl_seconds:
            self._hydrate()

    def top_k(self, query_embedding: List[float], k: int) -> List[PageNode]:
        self._ensure_ready()
        query_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_vec)) or 1.0

        scored: List[Tuple[float, PageNode]] = []
        for node in self._nodes:
            score = float(np.dot(query_vec, node.embedding) / (q_norm * node.norm))
            scored.append((score, node))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = []
        for rank, (score, node) in enumerate(scored[:k], start=1):
            clone = PageNode(
                **{**node.__dict__, "score": score},
            )
            top.append(clone)
        return top


index = SupabaseIndex()
