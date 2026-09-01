"""RAG 索引编排。"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy.orm import Session

from myfitness.rag.chunking import collect_chunks
from myfitness.rag.pgvector_setup import ensure_rag_schema, rag_is_available
from myfitness.rag.store import upsert_chunks

logger = logging.getLogger(__name__)


def index_user_data(
    session: Session,
    user_id: int,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    full: bool = False,
) -> dict:
    """索引用户数据到 pgvector。

    full=True 时忽略日期范围，重建全部 chunk。
    """
    bind = session.get_bind()
    if bind is not None:
        ensure_rag_schema(bind)

    if not rag_is_available(session):
        return {
            "status": "skipped",
            "reason": "RAG 不可用（需 PostgreSQL + pgvector + Embedding 配置）",
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
        }

    if full:
        start_date = None
        end_date = None

    documents = collect_chunks(
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
    )
    stats = upsert_chunks(session, user_id, documents)
    stats["status"] = "success"
    stats["total_documents"] = len(documents)
    logger.info(
        "RAG 索引完成 user=%s range=%s~%s stats=%s",
        user_id,
        start_date,
        end_date,
        stats,
    )
    return stats
