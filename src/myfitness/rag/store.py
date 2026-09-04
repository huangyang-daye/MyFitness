"""pgvector 向量块存储与双路检索。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.db.models import RagChunk
from myfitness.rag.bm25 import bm25_document_text
from myfitness.rag.chunking import content_hash
from myfitness.rag.dimensions import ensure_embedding_column_dimensions
from myfitness.rag.embedding import embed_texts
from myfitness.rag.hybrid import hybrid_rank, vector_knn
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


@dataclass(frozen=True)
class _SearchRow:
    id: int
    source_type: str
    source_id: str
    domain: str
    title: str
    content: str
    record_date: date | None
    chunk_metadata: dict[str, Any] | None
    embedding: list[float] | None = None


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
    """双路召回（向量 KNN + BM25）→ RRF → BM25 精排 → minScore 过滤。"""
    if not query.strip():
        return []

    settings = get_settings()
    if not settings.rag_enabled:
        return []

    top_k = top_k or settings.rag_top_k
    min_similarity = (
        settings.rag_min_similarity if min_similarity is None else min_similarity
    )
    recall_k = max(settings.rag_recall_k, top_k)

    rows = _load_search_rows(
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
        domain=domain,
    )
    if not rows:
        return []

    documents = [
        (row.id, bm25_document_text(row.title, row.content)) for row in rows
    ]
    vector_hits = _vector_recall(
        session,
        query.strip(),
        rows,
        recall_k=recall_k,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        domain=domain,
    )
    ranked = hybrid_rank(
        query.strip(),
        documents,
        vector_hits,
        recall_k=recall_k,
        top_k=top_k,
        rrf_k=settings.rag_rrf_k,
        min_score=min_similarity,
    )
    if not ranked:
        return []

    by_id = {row.id: row for row in rows}
    results: list[RetrievedChunk] = []
    for hit in ranked:
        row = by_id.get(hit.doc_id)
        if row is None:
            continue
        metadata = dict(row.chunk_metadata or {})
        metadata["vector_score"] = hit.vector_score
        metadata["bm25_score"] = hit.bm25_score
        metadata["rrf_score"] = hit.rrf_score
        results.append(
            RetrievedChunk(
                id=row.id,
                source_type=row.source_type,
                source_id=row.source_id,
                domain=row.domain,
                title=row.title,
                content=row.content,
                record_date=row.record_date,
                similarity=hit.score,
                metadata=metadata,
            )
        )
    return results


def _load_search_rows(
    session: Session,
    user_id: int,
    *,
    start_date: date | None,
    end_date: date | None,
    domain: str | None,
) -> list[_SearchRow]:
    bind = session.get_bind()
    load_embedding = bind is None or not is_postgresql(bind)
    columns = [
        RagChunk.id,
        RagChunk.source_type,
        RagChunk.source_id,
        RagChunk.domain,
        RagChunk.title,
        RagChunk.content,
        RagChunk.record_date,
        RagChunk.chunk_metadata,
    ]
    if load_embedding:
        columns.append(RagChunk.embedding)

    stmt = select(*columns).where(RagChunk.user_id == user_id)
    if start_date is not None:
        stmt = stmt.where(RagChunk.record_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(RagChunk.record_date <= end_date)
    if domain:
        stmt = stmt.where(RagChunk.domain == domain)

    rows: list[_SearchRow] = []
    for item in session.execute(stmt).mappings():
        embedding = item.get("embedding") if load_embedding else None
        if embedding is not None and not isinstance(embedding, list):
            embedding = list(embedding)
        rows.append(
            _SearchRow(
                id=int(item["id"]),
                source_type=str(item["source_type"]),
                source_id=str(item["source_id"]),
                domain=str(item["domain"]),
                title=str(item["title"] or ""),
                content=str(item["content"] or ""),
                record_date=item["record_date"],
                chunk_metadata=item["chunk_metadata"],
                embedding=embedding,
            )
        )
    return rows


def _vector_recall(
    session: Session,
    query: str,
    rows: list[_SearchRow],
    *,
    recall_k: int,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
    domain: str | None,
) -> list[tuple[int, float]]:
    from myfitness.rag.embedding import EmbeddingError, embed_text

    bind = session.get_bind()
    if bind is not None and is_postgresql(bind) and rag_is_available(session):
        try:
            query_vector = embed_text(query)
        except EmbeddingError as exc:
            logger.warning("向量召回跳过：%s", exc)
            return []
        return _vector_knn_sql(
            session,
            query_vector,
            recall_k=recall_k,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            domain=domain,
        )

    corpus = [
        (row.id, row.embedding)
        for row in rows
        if isinstance(row.embedding, list) and row.embedding
    ]
    if not corpus:
        return []
    try:
        query_vector = embed_text(query)
    except EmbeddingError as exc:
        logger.debug("向量召回跳过（无 embedding）：%s", exc)
        return []
    return vector_knn(query_vector, corpus, recall_k)


def _vector_knn_sql(
    session: Session,
    query_vector: list[float],
    *,
    recall_k: int,
    user_id: int,
    start_date: date | None,
    end_date: date | None,
    domain: str | None,
) -> list[tuple[int, float]]:
    filters = ["user_id = :user_id", "embedding IS NOT NULL"]
    params: dict = {
        "user_id": user_id,
        "query_vec": _vector_literal(query_vector),
        "top_k": recall_k,
    }
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
            1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM rag_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
        """
    )
    from myfitness.db.sql_logging import log_raw_sql

    log_raw_sql(str(sql), params)
    hits: list[tuple[int, float]] = []
    for row in session.execute(sql, params).mappings():
        similarity = float(row["similarity"] or 0.0)
        if similarity > 0:
            hits.append((int(row["id"]), similarity))
    return hits


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
