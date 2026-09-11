"""记忆管理器 — 工作记忆 / 情景记忆 / 用户画像 三级入口。"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.memory.episodic import load_episodic_text, persist_overflow_episode
from myfitness.memory.long_term import load_profile, update_long_term_from_message
from myfitness.memory.profile import profile_to_markdown
from myfitness.memory.short_term import split_working_window
from myfitness.memory.types import MemoryBundle
from myfitness.memory.working import format_working_text, load_working, save_working
from myfitness.schemas.state import ContextSnapshot, Intent, MyFitnessGraphState

logger = logging.getLogger(__name__)


def apply_memory_for_turn(
    session: Session,
    state: MyFitnessGraphState,
    *,
    intent: Intent | None = None,
) -> MemoryBundle:
    """每轮对话调用一次：刷新 Redis 工作记忆、落库情景摘要、更新画像。"""
    settings = get_settings()
    if not settings.memory_enabled:
        return MemoryBundle()

    try:
        working_text, compressed, backend = _refresh_working_and_episodic(session, state)
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
        episodic = load_episodic_text(
            session,
            state.user_id,
            state.session_id,
            fallback=state.session_memory,
        )
        return MemoryBundle(
            short_term=working_text,
            episodic=episodic,
            long_term=long_term,
            profile=profile,
            updated=updated,
            compressed=compressed,
            working_backend=backend,
        )
    except Exception as exc:  # noqa: BLE001 - 记忆失败不得打断主对话
        logger.warning("记忆系统本轮跳过: %s", exc)
        return MemoryBundle()


def attach_memory(context: ContextSnapshot, bundle: MemoryBundle) -> ContextSnapshot:
    return context.model_copy(
        update={
            "memory_short_term": bundle.short_term,
            "memory_episodic": bundle.episodic,
            "memory_long_term": bundle.long_term,
            "user_profile": bundle.profile,
        }
    )


def _refresh_working_and_episodic(
    session: Session,
    state: MyFitnessGraphState,
) -> tuple[str, bool, str]:
    settings = get_settings()
    snapshot = load_working(state.user_id, state.session_id)
    if snapshot is not None:
        state.memory_compacted_count = max(state.memory_compacted_count, snapshot.compacted_count)

    overflow, recent, new_compacted = split_working_window(
        list(state.messages),
        state.memory_compacted_count,
        settings.memory_short_term_turns,
    )
    compressed = False
    if overflow:
        persist_overflow_episode(
            session,
            state,
            overflow,
            turn_start=state.memory_compacted_count,
            turn_end=new_compacted,
            use_llm=True,
        )
        state.memory_compacted_count = new_compacted
        compressed = True

    backend = save_working(
        state.user_id,
        state.session_id,
        compacted_count=state.memory_compacted_count,
        recent=recent,
    )
    return format_working_text(recent), compressed, backend
