"""上下文压缩 — LLM 优先，失败回退规则摘要。"""

from __future__ import annotations

import logging

from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.schemas.state import ChatMessage

logger = logging.getLogger(__name__)

_ROLE_LABEL = {"user": "用户", "assistant": "助手"}


def format_messages(messages: list[ChatMessage], *, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for message in messages:
        role = _ROLE_LABEL.get(message.role, message.role)
        text = " ".join(message.content.strip().split())
        if len(text) > 400:
            text = text[:397] + "…"
        if text:
            lines.append(f"{role}：{text}")
    blob = "\n".join(lines)
    if len(blob) > max_chars:
        blob = blob[: max_chars - 1].rstrip() + "…"
    return blob


def compress_dialogue(
    overflow: list[ChatMessage],
    *,
    prior_summary: str = "",
    max_chars: int = 1200,
    use_llm: bool = True,
) -> str:
    """把较早对话压成短摘要，保留目标、偏好和未完成事项。"""
    transcript = format_messages(overflow)
    if not transcript and not prior_summary.strip():
        return ""
    if use_llm and is_llm_configured() and transcript:
        try:
            summary = _llm_compress(prior_summary, transcript, max_chars)
            if summary:
                return _clip(summary, max_chars)
        except Exception as exc:  # noqa: BLE001 - 压缩失败不影响主对话
            logger.warning("对话压缩 LLM 失败，改用规则摘要: %s", exc)
    return _rule_compress(prior_summary, transcript, max_chars)


def _llm_compress(prior: str, transcript: str, max_chars: int) -> str:
    prior_block = f"已有摘要：\n{prior.strip()}\n\n" if prior.strip() else ""
    messages = [
        {
            "role": "system",
            "content": (
                "你是健身助手的记忆压缩器。把对话压成简洁中文要点，"
                f"不超过 {max_chars} 字。保留：用户目标、饮食/训练偏好、伤病限制、未完成事项。"
                "不要编造；不要输出 JSON 或标题以外的客套话。"
            ),
        },
        {
            "role": "user",
            "content": f"{prior_block}需要压缩的对话：\n{transcript}",
        },
    ]
    return chat_completion(messages, temperature=0.0, max_tokens=500).strip()


def _rule_compress(prior: str, transcript: str, max_chars: int) -> str:
    parts: list[str] = []
    if prior.strip():
        parts.append(prior.strip())
    if transcript.strip():
        parts.append(transcript.strip())
    merged = "\n".join(parts)
    return _clip(merged, max_chars)


def _clip(text: str, max_chars: int) -> str:
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"
