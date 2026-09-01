"""短期记忆 — 保留最近若干轮，把更早的对话压缩进 session_memory。"""

from __future__ import annotations

from myfitness.config import get_settings
from myfitness.memory.compress import compress_dialogue, format_messages
from myfitness.schemas.state import MyFitnessGraphState


def build_short_term(state: MyFitnessGraphState, *, use_llm: bool = True) -> tuple[str, bool]:
    """更新 state.session_memory，返回注入 Prompt 的短期记忆文本。"""
    settings = get_settings()
    keep = settings.memory_short_term_turns
    messages = list(state.messages)
    compacted = max(0, min(state.memory_compacted_count, len(messages)))
    compressed = False

    overflow_end = len(messages) - keep
    if overflow_end > compacted:
        overflow = messages[compacted:overflow_end]
        state.session_memory = compress_dialogue(
            overflow,
            prior_summary=state.session_memory,
            max_chars=settings.memory_compress_chars,
            use_llm=use_llm,
        )
        state.memory_compacted_count = overflow_end
        compressed = True

    recent = messages[-keep:] if messages else []
    parts: list[str] = []
    if state.session_memory.strip():
        parts.append("较早对话摘要：\n" + state.session_memory.strip())
    if recent:
        parts.append("最近对话：\n" + format_messages(recent, max_chars=3000))
    return "\n\n".join(parts).strip(), compressed
