"""pgvector embedding 列维度 — 与 EMBEDDING_DIMENSIONS 对齐。"""

from __future__ import annotations

import logging
import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

from myfitness.config import get_settings

logger = logging.getLogger(__name__)

_VECTOR_DIM_RE = re.compile(r"vector\((\d+)\)")


def _is_postgresql(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def parse_vector_column_type(type_spec: str | None) -> int | None:
    if not type_spec:
        return None
    match = _VECTOR_DIM_RE.search(type_spec.strip().lower())
    if not match:
        return None
    return int(match.group(1))


def read_embedding_column_dimensions(engine: Engine) -> int | None:
    """读取 rag_chunks.embedding 列维度；表或列不存在时返回 None。"""
    if not _is_postgresql(engine):
        return None
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod) AS col_type
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = current_schema()
                  AND c.relname = 'rag_chunks'
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
        ).scalar()
    return parse_vector_column_type(str(row) if row else None)


def ensure_embedding_column_dimensions(engine: Engine) -> bool:
    """将 embedding 列调整为 settings.embedding_dimensions；必要时清空旧向量。"""
    if not _is_postgresql(engine):
        return True

    expected = get_settings().embedding_dimensions
    current = read_embedding_column_dimensions(engine)
    if current is None or current == expected:
        return True

    logger.warning(
        "rag_chunks.embedding 为 vector(%s)，与 EMBEDDING_DIMENSIONS=%s 不一致；"
        "将清空已有向量并调整列类型",
        current,
        expected,
    )
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_rag_chunks_embedding_hnsw"))
        conn.execute(text("UPDATE rag_chunks SET embedding = NULL"))
        conn.execute(
            text(f"ALTER TABLE rag_chunks ALTER COLUMN embedding TYPE vector({expected})")
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
    return True


def embedding_dimensions_compatible(engine: Engine) -> bool:
    current = read_embedding_column_dimensions(engine)
    if current is None:
        return True
    return current == get_settings().embedding_dimensions
