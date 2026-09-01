"""从用户提问抽取/合并长期画像。"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from myfitness.config import get_settings
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.memory.types import PROFILE_KEYS, PROFILE_LABELS
from myfitness.schemas.state import Intent

logger = logging.getLogger(__name__)

_SKIP_INTENTS = {Intent.CONFIRMATION_RESPONSE}
_SKIP_MESSAGES = {"你好", "您好", "谢谢", "感谢", "确认", "取消", "ok", "OK", "是的", "好的"}

_GOAL_PATTERNS = (
    re.compile(r"目标体重\s*(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤)?", re.IGNORECASE),
    re.compile(r"目标体脂\s*(?:是|为)?\s*(\d+(?:\.\d+)?)\s*%?"),
    re.compile(r"(减脂|增肌|减重|增重|塑形|维持体重)"),
)
_DIET_PATTERNS = (
    re.compile(r"(不吃[\w\u4e00-\u9fff]{1,8}|忌口[\w\u4e00-\u9fff]{1,8})"),
    re.compile(r"(乳糖不耐|过敏|素食|纯素|低碳|生酮|少油|高蛋白|控制碳水)"),
)
_TRAINING_PATTERNS = (
    re.compile(r"(膝盖(?:伤|疼|不适)?|肩伤|腰痛|腕伤)"),
    re.compile(r"(只做有氧|力量训练|练胸|练背|练腿|卧推|深蹲|硬拉)"),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"(喜欢|偏好|习惯).{0,12}(图表|折线|日报|晨报|蛋白质|鸡胸)"),
)


def should_extract(message: str, intent: Intent | None) -> bool:
    text = message.strip()
    if len(text) < 2 or text in _SKIP_MESSAGES:
        return False
    return intent not in _SKIP_INTENTS


def extract_profile_facts(
    message: str,
    *,
    intent: Intent | None = None,
    existing: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, list[str]]:
    """返回各字段新增条目（可能为空）。"""
    if not should_extract(message, intent):
        return {}
    facts = _rule_extract(message, intent)
    if use_llm and is_llm_configured():
        try:
            llm_facts = _llm_extract(message, existing or {}, intent)
            facts = _merge_fact_maps(facts, llm_facts)
        except Exception as exc:  # noqa: BLE001 - 画像抽取失败不影响对话
            logger.warning("用户画像 LLM 抽取失败，使用规则结果: %s", exc)
    return {key: values for key, values in facts.items() if values}


def merge_profile(
    existing: dict[str, Any] | None,
    incoming: dict[str, list[str]],
    *,
    max_items: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    limit = max_items or settings.memory_profile_max_items
    merged: dict[str, Any] = {key: [] for key in PROFILE_KEYS}
    source = existing or {}
    for key in PROFILE_KEYS:
        merged[key] = _merge_items(_as_str_list(source.get(key)), incoming.get(key, []), limit)
    merged["updated_at"] = datetime.now(UTC).isoformat()
    return merged


def profile_to_markdown(profile: dict[str, Any]) -> str:
    lines = ["# 用户画像", ""]
    has_item = False
    for key in PROFILE_KEYS:
        items = _as_str_list(profile.get(key))
        if not items:
            continue
        has_item = True
        lines.append(f"## {PROFILE_LABELS[key]}")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    if not has_item:
        return ""
    updated = profile.get("updated_at")
    if updated:
        lines.append(f"_更新于 {updated}_")
    return "\n".join(lines).strip()


def _rule_extract(message: str, intent: Intent | None) -> dict[str, list[str]]:
    facts: dict[str, list[str]] = {key: [] for key in PROFILE_KEYS}
    for pattern in _GOAL_PATTERNS:
        match = pattern.search(message)
        if match:
            snippet = match.group(0).strip()
            if match.lastindex:
                snippet = match.group(0).strip()
            facts["goals"].append(snippet)
    for pattern in _DIET_PATTERNS:
        match = pattern.search(message)
        if match:
            facts["diet"].append(match.group(1).strip())
    for pattern in _TRAINING_PATTERNS:
        match = pattern.search(message)
        if match:
            facts["training"].append(match.group(1).strip())
    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(message)
        if match:
            facts["preferences"].append(match.group(0).strip())
    if "伤" in message or "疼" in message or "不适" in message:
        facts["constraints"].extend(
            item for item in facts["training"] if any(k in item for k in ("伤", "疼", "不适"))
        )
    if intent in {Intent.DATA_QUERY, Intent.TREND_ANALYSIS, Intent.GOAL_SETTING}:
        compact = " ".join(message.strip().split())
        if 4 <= len(compact) <= 40:
            facts["habits"].append(compact)
    return facts


def _llm_extract(
    message: str,
    existing: dict[str, Any],
    intent: Intent | None,
) -> dict[str, list[str]]:
    existing_md = profile_to_markdown(existing) or "（尚无画像）"
    intent_label = intent.value if intent else "unknown"
    prompt = (
        "根据用户最新提问，抽取应写入长期画像的事实。"
        "只输出 JSON 对象，键为 goals/diet/training/constraints/preferences/habits，值为字符串数组。"
        "没有新事实则对应数组为空。不要重复已有画像中的相同内容。\n\n"
        f"已有画像：\n{existing_md}\n\n"
        f"意图：{intent_label}\n用户提问：{message}"
    )
    raw = chat_completion(
        [
            {
                "role": "system",
                "content": "你是健身助手的画像抽取器。只输出 JSON，不要 Markdown。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    parsed = _extract_json(raw) or {}
    facts: dict[str, list[str]] = {}
    for key in PROFILE_KEYS:
        facts[key] = _as_str_list(parsed.get(key))
    return facts


def _merge_fact_maps(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for key in PROFILE_KEYS:
        merged[key] = _merge_items(left.get(key, []), right.get(key, []), 20)
    return merged


def _merge_items(old: list[str], new: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: list[str] = []
    for item in [*old, *new]:
        text = _clean_item(item)
        if not text:
            continue
        key = text.casefold()
        if any(key == prev or key in prev or prev in key for prev in seen):
            continue
        seen.append(key)
        result.append(text)
    return result[-limit:]


def _clean_item(value: str) -> str:
    text = " ".join(str(value).split())
    if len(text) > 80:
        text = text[:77] + "…"
    return text


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _extract_json(content: str) -> dict | None:
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
