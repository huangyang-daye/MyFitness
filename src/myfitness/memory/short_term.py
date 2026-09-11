"""工作记忆窗口 — 保留最近若干轮，溢出部分交给情景记忆层。"""

from __future__ import annotations

from myfitness.config import get_settings
from myfitness.memory.compress import clip_text, compress_dialogue
from myfitness.memory.working import format_working_text
from myfitness.schemas.state import ChatMessage, MyFitnessGraphState


def split_working_window(
    messages: list[ChatMessage],
    compacted_count: int,
    keep: int,
) -> tuple[list[ChatMessage], list[ChatMessage], int]:
    """返回 (overflow, recent, new_compacted_count)。"""
    compacted = max(0, min(compacted_count, len(messages)))
    overflow_end = max(compacted, len(messages) - keep)
    if overflow_end <= compacted:
        recent = messages[-keep:] if messages else []
        return [], recent, compacted
    overflow = messages[compacted:overflow_end]
    recent = messages[-keep:] if messages else []
    return overflow, recent, overflow_end


def build_short_term(state: MyFitnessGraphState, *, use_llm: bool = True) -> tuple[str, bool]:
    """无数据库时的工作记忆窗口：压缩溢出到 session_memory，返回最近对话文本。"""
    settings = get_settings()
    overflow, recent, new_compacted = split_working_window(
        list(state.messages),
        state.memory_compacted_count,
        settings.memory_short_term_turns,
    )
    compressed = bool(overflow)
    if compressed:
        batch = compress_dialogue(
            overflow,
            prior_summary="",
            max_chars=settings.memory_compress_chars,
            use_llm=use_llm,
        ).strip()
        if batch:
            merged = "\n".join(part for part in (state.session_memory.strip(), batch) if part)
            state.session_memory = clip_text(merged, settings.memory_compress_chars)
        state.memory_compacted_count = new_compacted
    return format_working_text(recent), compressed
