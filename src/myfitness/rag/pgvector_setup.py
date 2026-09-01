"""pgvector 扩展与 RAG 表初始化。"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.db.models import Base
from myfitness.rag.dimensions import ensure_embedding_column_dimensions

logger = logging.getLogger(__name__)


def is_postgresql(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


_PGVECTOR_INSTALL_HINT = (
    "无法在当前 PostgreSQL 上启用 vector 扩展。"
    "MyFitness 复用已有数据库实例，不要另起占用 5432 的 pgvector 容器。"
    "请把 pgvector 装进现有 Postgres（可用 docker/postgres/Dockerfile 基于 postgres:15-alpine 构建）。"
)


def ensure_pgvector_extension(engine: Engine) -> bool:
    """创建 pgvector 扩展；非 PostgreSQL 返回 False。"""
    if not is_postgresql(engine):
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        raise RuntimeError(_PGVECTOR_INSTALL_HINT) from exc
    return True


def ensure_rag_schema(engine: Engine) -> None:
    """创建 RAG 表与向量索引（PostgreSQL）。"""
    ensure_pgvector_extension(engine)
    tables = [
        Base.metadata.tables["rag_chunks"],
        Base.metadata.tables["knowledge_entries"],
    ]
    Base.metadata.create_all(engine, tables=tables)
    if not is_postgresql(engine):
        return
    ensure_embedding_column_dimensions(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
        )


def rag_is_available(session: Session) -> bool:
    """RAG 是否可用：PostgreSQL + pgvector + embedding 配置 + 列维度一致。"""
    from myfitness.rag.dimensions import embedding_dimensions_compatible
    from myfitness.rag.embedding import is_embedding_configured

    settings = get_settings()
    if not settings.rag_enabled:
        return False
    bind = session.get_bind()
    if bind is None or not is_postgresql(bind):
        return False
    if not is_embedding_configured():
        return False
    try:
        found = session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if found is None:
            return False
        if not embedding_dimensions_compatible(bind):
            try:
                ensure_embedding_column_dimensions(bind)
            except Exception:  # noqa: BLE001 - auto-migrate best effort
                logger.warning("无法自动调整 rag_chunks.embedding 列维度", exc_info=True)
                return False
            if not embedding_dimensions_compatible(bind):
                return False
        return True
    except Exception:  # noqa: BLE001 - dialect / permission differences
        return False
