"""知识库 CRUD 与向量索引。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from myfitness.db.models import KnowledgeEntry
from myfitness.db.repositories.knowledge import KnowledgeRepository
from myfitness.rag.chunking import entry_to_chunks
from myfitness.rag.embedding import is_embedding_configured
from myfitness.rag.indexer import index_user_data
from myfitness.rag.pgvector_setup import rag_is_available
from myfitness.rag.store import chunk_stats, count_chunks, delete_knowledge_chunks, upsert_chunks

logger = logging.getLogger(__name__)

MAX_TITLE_LEN = 200
MAX_CONTENT_LEN = 50_000


class KnowledgeError(ValueError):
    pass


class KnowledgeNotFound(KnowledgeError):
    pass


def _validate_title(title: str) -> str:
    value = title.strip()
    if not value:
        raise KnowledgeError("标题不能为空")
    if len(value) > MAX_TITLE_LEN:
        raise KnowledgeError(f"标题不能超过 {MAX_TITLE_LEN} 个字符")
    return value


def _validate_content(content: str) -> str:
    value = content.strip()
    if not value:
        raise KnowledgeError("内容不能为空")
    if len(value) > MAX_CONTENT_LEN:
        raise KnowledgeError(f"内容不能超过 {MAX_CONTENT_LEN} 个字符")
    return value


def entry_payload(entry: KnowledgeEntry) -> dict:
    updated = entry.updated_at or entry.created_at
    preview = entry.content.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:117] + "…"
    return {
        "id": entry.id,
        "title": entry.title,
        "content": entry.content,
        "preview": preview,
        "kind": entry.kind or "user",
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": updated.isoformat() if updated else None,
    }


def knowledge_overview(session: Session, user_id: int) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entries = repo.list_all()
    stats = chunk_stats(session, user_id)
    return {
        "available": rag_is_available(session),
        "embedding_configured": is_embedding_configured(),
        "entry_count": len(entries),
        "indexed_chunks": stats.get("knowledge", 0),
        "total_chunks": count_chunks(session, user_id),
        "chunk_stats": stats,
    }


def list_knowledge(session: Session, user_id: int) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entries = repo.list_all()
    return {
        **knowledge_overview(session, user_id),
        "entries": [entry_payload(item) for item in entries],
    }


def create_knowledge(session: Session, user_id: int, *, title: str, content: str) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entry = repo.create(_validate_title(title), _validate_content(content))
    index_result = index_knowledge_entry(session, user_id, entry)
    return {"entry": entry_payload(entry), "index": index_result}


def update_knowledge(
    session: Session,
    user_id: int,
    entry_id: int,
    *,
    title: str | None = None,
    content: str | None = None,
) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entry = repo.get(entry_id)
    if entry is None:
        raise KnowledgeNotFound(f"未找到知识条目：{entry_id}")

    if title is not None:
        entry.title = _validate_title(title)
    if content is not None:
        entry.content = _validate_content(content)
    entry.updated_at = datetime.now(UTC)
    session.flush()

    delete_knowledge_chunks(session, user_id, entry.id)
    index_result = index_knowledge_entry(session, user_id, entry)
    return {"entry": entry_payload(entry), "index": index_result}


def delete_knowledge(session: Session, user_id: int, entry_id: int) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entry = repo.get(entry_id)
    if entry is None:
        raise KnowledgeNotFound(f"未找到知识条目：{entry_id}")
    delete_knowledge_chunks(session, user_id, entry.id)
    repo.delete(entry)
    return {"deleted_id": entry_id}


def index_knowledge_entry(session: Session, user_id: int, entry: KnowledgeEntry) -> dict:
    documents = entry_to_chunks(entry)
    return upsert_chunks(session, user_id, documents)


def reindex_all_knowledge(session: Session, user_id: int) -> dict:
    repo = KnowledgeRepository(session, user_id)
    entries = repo.list_all()
    indexed = skipped = failed = 0
    for entry in entries:
        delete_knowledge_chunks(session, user_id, entry.id)
        stats = index_knowledge_entry(session, user_id, entry)
        indexed += stats.get("indexed", 0)
        skipped += stats.get("skipped", 0)
        failed += stats.get("failed", 0)
    return {
        "status": "success",
        "entries": len(entries),
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
    }


def reindex_all_data(session: Session, user_id: int) -> dict:
    knowledge = reindex_all_knowledge(session, user_id)
    auto = index_user_data(session, user_id, full=True)
    return {"knowledge": knowledge, "auto": auto}
