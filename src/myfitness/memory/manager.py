"""记忆管理器 — 短期窗口 + 长期画像 + 上下文压缩的统一入口。"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.memory.long_term import load_profile, update_long_term_from_message
from myfitness.memory.profile import profile_to_markdown
from myfitness.memory.short_term import build_short_term
from myfitness.memory.types import MemoryBundle
from myfitness.schemas.state import ContextSnapshot, Intent, MyFitnessGraphState

logger = logging.getLogger(__name__)


def apply_memory_for_turn(
    session: Session,
    state: MyFitnessGraphState,
    *,
    intent: Intent | None = None,
) -> MemoryBundle:
    """每轮对话调用一次：压缩短期记忆，并按提问更新长期画像。"""
    settings = get_settings()
    if not settings.memory_enabled:
        return MemoryBundle()

    try:
        short_term, compressed = build_short_term(state, use_llm=True)
        profile, long_term, updated = update_long_term_from_message(
            session,
            state.user_id,
            state.user_message,
            intent=intent,
            use_llm=True,
        )
        if not long_term:
            profile = load_profile(session, state.user_id)
            long_term = profile_to_markdown(profile)
        return MemoryBundle(
            short_term=short_term,
            long_term=long_term,
            profile=profile,
            updated=updated,
            compressed=compressed,
        )
    except Exception as exc:  # noqa: BLE001 - 记忆失败不得打断主对话
        logger.warning("记忆系统本轮跳过: %s", exc)
        return MemoryBundle()


def attach_memory(context: ContextSnapshot, bundle: MemoryBundle) -> ContextSnapshot:
    return context.model_copy(
        update={
            "memory_short_term": bundle.short_term,
            "memory_long_term": bundle.long_term,
            "user_profile": bundle.profile,
        }
    )
