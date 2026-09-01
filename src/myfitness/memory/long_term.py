"""长期记忆 — 用户画像写入 User.profile 与知识库。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from myfitness.db.models import KnowledgeEntry, User
from myfitness.db.repositories.knowledge import KnowledgeRepository
from myfitness.memory.profile import extract_profile_facts, merge_profile, profile_to_markdown
from myfitness.memory.types import PROFILE_TITLE
from myfitness.rag.knowledge_service import index_knowledge_entry
from myfitness.schemas.state import Intent

logger = logging.getLogger(__name__)


def load_profile(session: Session, user_id: int) -> dict[str, Any]:
    user = session.get(User, user_id)
    if user is None or not isinstance(user.profile, dict):
        return {}
    return dict(user.profile)


def update_long_term_from_message(
    session: Session,
    user_id: int,
    message: str,
    *,
    intent: Intent | None = None,
    use_llm: bool = True,
) -> tuple[dict[str, Any], str, bool]:
    """根据本轮提问更新画像；返回 (profile, markdown, changed)。"""
    existing = load_profile(session, user_id)
    incoming = extract_profile_facts(
        message, intent=intent, existing=existing, use_llm=use_llm
    )
    if not incoming:
        markdown = profile_to_markdown(existing)
        return existing, markdown, False

    merged = merge_profile(existing, incoming)
    if _profiles_equivalent(existing, merged):
        return existing, profile_to_markdown(existing), False

    _save_user_profile(session, user_id, merged)
    markdown = profile_to_markdown(merged)
    if markdown:
        _upsert_knowledge(session, user_id, markdown)
    return merged, markdown, True


def _save_user_profile(session: Session, user_id: int, profile: dict[str, Any]) -> None:
    user = session.get(User, user_id)
    if user is None:
        return
    user.profile = profile
    user.updated_at = datetime.now(UTC)
    session.flush()


def _upsert_knowledge(session: Session, user_id: int, content: str) -> KnowledgeEntry | None:
    repo = KnowledgeRepository(session, user_id)
    entry = repo.upsert_memory(PROFILE_TITLE, content)
    try:
        index_knowledge_entry(session, user_id, entry)
    except Exception as exc:  # noqa: BLE001 - 向量索引失败仍保留知识库文本
        logger.warning("长期记忆写入知识库成功，但向量索引跳过: %s", exc)
    return entry


def _profiles_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    from myfitness.memory.types import PROFILE_KEYS

    for key in PROFILE_KEYS:
        if [str(item) for item in (left.get(key) or [])] != [
            str(item) for item in (right.get(key) or [])
        ]:
            return False
    return True
