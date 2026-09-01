"""pgvector 向量块存储与检索。"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.db.models import RagChunk
from myfitness.rag.chunking import content_hash
from myfitness.rag.dimensions import ensure_embedding_column_dimensions
from myfitness.rag.embedding import embed_texts
from myfitness.rag.pgvector_setup import is_postgresql, rag_is_available
from myfitness.rag.schemas import ChunkDocument, RetrievedChunk

logger = logging.getLogger(__name__)


def upsert_chunks(session: Session, user_id: int, documents: list[ChunkDocument]) -> dict[str, int]:
    """写入或更新向量块；内容未变则跳过 re-embed。"""
    if not documents:
        return {"indexed": 0, "skipped": 0, "failed": 0}

    bind = session.get_bind()
    if bind is not None and is_postgresql(bind):
        ensure_embedding_column_dimensions(bind)

    if not rag_is_available(session):
        logger.info("RAG 不可用，跳过索引")
        return {"indexed": 0, "skipped": len(documents), "failed": 0}

    settings = get_settings()
    expected_dims = settings.embedding_dimensions
    indexed = skipped = failed = 0
    batch_size = settings.rag_index_batch_size

    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        to_embed: list[ChunkDocument] = []
        for doc in batch:
            existing = session.scalar(
                select(RagChunk).where(
                    RagChunk.user_id == user_id,
                    RagChunk.source_type == doc.source_type,
                    RagChunk.source_id == doc.source_id,
                )
            )
            digest = content_hash(doc.content)
            if existing and existing.content_hash == digest and existing.embedding is not None:
                skipped += 1
                continue
            to_embed.append(doc)

        if not to_embed:
            continue

        try:
            vectors = embed_texts([doc.content for doc in to_embed])
        except Exception as exc:  # noqa: BLE001 - batch failure counts per doc
            logger.warning("Embedding 批次失败: %s", exc)
            failed += len(to_embed)
            continue

        batch_indexed = 0
        for doc, vector in zip(to_embed, vectors, strict=True):
            if len(vector) != expected_dims:
                logger.warning(
                    "跳过向量块 %s/%s：维度 %s 与 EMBEDDING_DIMENSIONS=%s 不一致",
                    doc.source_type,
                    doc.source_id,
                    len(vector),
                    expected_dims,
                )
                failed += 1
                continue

            digest = content_hash(doc.content)
            row = session.scalar(
                select(RagChunk).where(
                    RagChunk.user_id == user_id,
                    RagChunk.source_type == doc.source_type,
                    RagChunk.source_id == doc.source_id,
                )
            )
            if row is None:
                row = RagChunk(
                    user_id=user_id,
                    source_type=doc.source_type,
                    source_id=doc.source_id,
                )
                session.add(row)

            row.domain = doc.domain
            row.record_date = doc.record_date
            row.title = doc.title
            row.content = doc.content
            row.content_hash = digest
            row.chunk_metadata = doc.metadata
            row.embedding = vector
            indexed += 1
            batch_indexed += 1

        if batch_indexed == 0:
            continue

        savepoint = session.begin_nested()
        try:
            session.flush()
        except Exception as exc:  # noqa: BLE001 - keep main transaction usable
            savepoint.rollback()
            logger.warning("向量块写入失败（已回滚本批次）: %s", exc)
            failed += batch_indexed
            indexed -= batch_indexed

    return {"indexed": indexed, "skipped": skipped, "failed": failed}


def search_chunks(
    session: Session,
    user_id: int,
    query: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    domain: str | None = None,
) -> list[RetrievedChunk]:
    """向量相似度检索。"""
    if not query.strip() or not rag_is_available(session):
        return []

    settings = get_settings()
    top_k = top_k or settings.rag_top_k
    min_similarity = (
        settings.rag_min_similarity if min_similarity is None else min_similarity
    )

    from myfitness.rag.embedding import EmbeddingError, embed_text

    try:
        query_vector = embed_text(query.strip())
    except EmbeddingError as exc:
        logger.warning("语义检索跳过：%s", exc)
        return []

    bind = session.get_bind()
    if bind is None or not is_postgresql(bind):
        return []

    vector_literal = _vector_literal(query_vector)
    filters = ["user_id = :user_id", "embedding IS NOT NULL"]
    params: dict = {"user_id": user_id, "query_vec": vector_literal, "top_k": top_k}

    if start_date is not None:
        filters.append("record_date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        filters.append("record_date <= :end_date")
        params["end_date"] = end_date
    if domain:
        filters.append("domain = :domain")
        params["domain"] = domain

    where_sql = " AND ".join(filters)
    sql = text(
        f"""
        SELECT
            id,
            source_type,
            source_id,
            domain,
            title,
            content,
            record_date,
            chunk_metadata,
            1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
        """
    )

    rows = session.execute(sql, params).mappings().all()
    results: list[RetrievedChunk] = []
    for row in rows:
        similarity = float(row["similarity"] or 0.0)
        if similarity < min_similarity:
            continue
        record_date = row["record_date"]
        results.append(
            RetrievedChunk(
                id=int(row["id"]),
                source_type=str(row["source_type"]),
                source_id=str(row["source_id"]),
                domain=str(row["domain"]),
                title=str(row["title"] or ""),
                content=str(row["content"] or ""),
                record_date=record_date,
                similarity=similarity,
                metadata=row["chunk_metadata"],
            )
        )
    return results


def delete_knowledge_chunks(session: Session, user_id: int, knowledge_id: int) -> int:
    pattern = f"{knowledge_id}:%"
    result = session.execute(
        delete(RagChunk).where(
            RagChunk.user_id == user_id,
            RagChunk.source_type == "knowledge",
            RagChunk.source_id.like(pattern),
        )
    )
    return int(result.rowcount or 0)


def delete_chunks_for_range(
    session: Session,
    user_id: int,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    stmt = delete(RagChunk).where(RagChunk.user_id == user_id)
    if start_date is not None:
        stmt = stmt.where(RagChunk.record_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(RagChunk.record_date <= end_date)
    result = session.execute(stmt)
    return int(result.rowcount or 0)


def count_chunks(session: Session, user_id: int) -> int:
    value = session.scalar(
        select(func.count()).select_from(RagChunk).where(RagChunk.user_id == user_id)
    )
    return int(value or 0)


def chunk_stats(session: Session, user_id: int) -> dict[str, int]:
    if not rag_is_available(session):
        return {}
    rows = session.execute(
        text(
            """
            SELECT source_type, COUNT(*) AS count
            FROM rag_chunks
            WHERE user_id = :user_id
            GROUP BY source_type
            ORDER BY source_type
            """
        ),
        {"user_id": user_id},
    ).mappings()
    return {str(row["source_type"]): int(row["count"]) for row in rows}


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
