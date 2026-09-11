"""向量库索引后台任务 — 同步 embedding / pgvector 写入放入线程池。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from myfitness.config import get_settings

logger = logging.getLogger(__name__)

_lock = Lock()
_executor: ThreadPoolExecutor | None = None


def schedule_memory_index(user_id: int, title: str, content: str) -> None:
    """主链路只投递任务，不在当前线程做 embedding / upsert。"""
    text = (content or "").strip()
    if not text:
        return
    _pool().submit(_run_memory_index, user_id, title, text)


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            workers = get_settings().memory_vector_pool_size
            _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mf-vec")
        return _executor


def _run_memory_index(user_id: int, title: str, content: str) -> None:
    try:
        from myfitness.db.repositories.knowledge import KnowledgeRepository
        from myfitness.db.session import session_scope
        from myfitness.rag.knowledge_service import index_knowledge_entry

        with session_scope() as session:
            entry = KnowledgeRepository(session, user_id).upsert_memory(title, content)
            index_knowledge_entry(session, user_id, entry)
    except Exception as exc:  # noqa: BLE001 - 向量索引失败不影响对话
        logger.warning("画像向量索引后台任务失败: %s", exc)
