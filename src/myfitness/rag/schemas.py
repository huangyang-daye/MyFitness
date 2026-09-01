"""RAG 公共 schema。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ChunkDocument:
    source_type: str
    source_id: str
    domain: str
    title: str
    content: str
    record_date: date | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    source_type: str
    source_id: str
    domain: str
    title: str
    content: str
    record_date: date | None
    similarity: float
    metadata: dict[str, Any] | None = None
