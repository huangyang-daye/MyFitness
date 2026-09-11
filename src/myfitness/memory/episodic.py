"""情景记忆 — 将溢出对话摘要写入 PostgreSQL，并组装 Prompt 文本。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from myfitness.config import get_settings
from myfitness.db.repositories.episodes import EpisodeRepository
from myfitness.memory.compress import clip_text, compress_dialogue
from myfitness.schemas.state import ChatMessage, MyFitnessGraphState


def persist_overflow_episode(
    session: Session,
    state: MyFitnessGraphState,
    overflow: list[ChatMessage],
    *,
    turn_start: int,
    turn_end: int,
    use_llm: bool = True,
) -> str:
    """压缩一批溢出对话：写入情景表，并更新滚动 session_memory。"""
    settings = get_settings()
    batch = compress_dialogue(
        overflow,
        prior_summary="",
        max_chars=settings.memory_compress_chars,
        use_llm=use_llm,
    ).strip()
    if not batch:
        return state.session_memory
    EpisodeRepository(session, state.user_id).add_if_absent(
        state.session_id,
        batch,
        turn_start=turn_start,
        turn_end=turn_end,
    )
    merged = "\n".join(part for part in (state.session_memory.strip(), batch) if part)
    state.session_memory = clip_text(merged, settings.memory_compress_chars)
    return batch


def load_episodic_text(
    session: Session,
    user_id: int,
    session_id: str,
    *,
    fallback: str = "",
) -> str:
    settings = get_settings()
    rows = EpisodeRepository(session, user_id).list_recent(limit=settings.memory_episodic_limit)
    if not rows:
        text = fallback.strip()
        return f"较早对话摘要：\n{text}" if text else ""
    lines = ["历史情景摘要："]
    for row in rows:
        label = "本会话" if row.session_id == session_id else f"会话 {row.session_id[:8]}"
        summary = " ".join(row.summary.split())
        if len(summary) > 400:
            summary = summary[:397] + "…"
        lines.append(f"- [{label}] {summary}")
    return "\n".join(lines)
