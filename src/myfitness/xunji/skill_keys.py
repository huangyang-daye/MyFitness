"""从 skills/xunji-*/SKILL.md 解析 Bearer Token。

训记 App 不再单独发放 API Key；用户从 App 复制 Skill 文档后，鉴权 Token 嵌入在
SKILL.md 的「鉴权」章节。本模块负责读取；.env 中的 XUNJI_* 变量仍可覆盖。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from myfitness.config import Settings, get_settings
from myfitness.xunji.skills import SKILL_BODY, SKILL_DOC_PATHS, SKILL_FOOD, SKILL_TRAINING

BEARER_INLINE_RE = re.compile(r"Bearer\s+([^\s`]+)")
TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "body": re.compile(r"xjbody_[a-f0-9]+"),
    "food": re.compile(r"xjfood_[a-f0-9]+"),
    "training": re.compile(r"xjllm_[a-f0-9]+"),
}

SKILL_NAME_BY_KEY = {
    "body": SKILL_BODY,
    "food": SKILL_FOOD,
    "training": SKILL_TRAINING,
}


def _read_skill_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_token(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(0) if match else ""


def _extract_food_search_key(text: str) -> str:
    for line in text.splitlines():
        if "食物搜索" in line and "Bearer" in line:
            match = BEARER_INLINE_RE.search(line)
            if match:
                return match.group(1)
    for match in BEARER_INLINE_RE.finditer(text):
        token = match.group(1)
        if not token.startswith("xjfood_"):
            return token
    return ""


@lru_cache
def load_keys_from_skills() -> dict[str, str]:
    """从项目内 Skill 文档解析 Token；缺失则对应值为空字符串。"""
    keys: dict[str, str] = {"body": "", "food": "", "food_search": "", "training": ""}

    body_path = SKILL_DOC_PATHS.get(SKILL_BODY)
    if body_path and body_path.exists():
        text = _read_skill_text(body_path)
        keys["body"] = _extract_token(text, TOKEN_PATTERNS["body"])

    food_path = SKILL_DOC_PATHS.get(SKILL_FOOD)
    if food_path and food_path.exists():
        text = _read_skill_text(food_path)
        keys["food"] = _extract_token(text, TOKEN_PATTERNS["food"])
        keys["food_search"] = _extract_food_search_key(text)

    training_path = SKILL_DOC_PATHS.get(SKILL_TRAINING)
    if training_path and training_path.exists():
        text = _read_skill_text(training_path)
        keys["training"] = _extract_token(text, TOKEN_PATTERNS["training"])

    return keys


def resolve_xunji_keys(settings: Settings | None = None) -> dict[str, str]:
    """合并 .env 与 Skill 文档：环境变量优先，否则用 Skill 内嵌 Token。"""
    s = settings or get_settings()
    skill = load_keys_from_skills()
    return {
        "body": (s.xunji_body_api_key or "").strip() or skill["body"],
        "food": (s.xunji_food_api_key or "").strip() or skill["food"],
        "food_search": (s.xunji_food_search_key or "").strip() or skill["food_search"],
        "training": (s.xunji_training_api_key or "").strip() or skill["training"],
    }


def key_source(name: str, settings: Settings | None = None) -> str:
    """返回 key 来源：env / skill / missing。"""
    s = settings or get_settings()
    env_field = {
        "body": "xunji_body_api_key",
        "food": "xunji_food_api_key",
        "food_search": "xunji_food_search_key",
        "training": "xunji_training_api_key",
    }[name]
    if getattr(s, env_field, "").strip():
        return "env"
    skill = load_keys_from_skills()
    if skill.get(name):
        return "skill"
    return "missing"
