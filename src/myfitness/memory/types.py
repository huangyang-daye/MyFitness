"""记忆系统公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROFILE_KEYS = ("goals", "diet", "training", "constraints", "preferences", "habits")
PROFILE_LABELS = {
    "goals": "目标",
    "diet": "饮食",
    "training": "训练",
    "constraints": "限制/伤病",
    "preferences": "偏好",
    "habits": "提问习惯",
}
PROFILE_TITLE = "【长期记忆】用户画像"
MEMORY_KIND = "memory"
USER_KIND = "user"


@dataclass
class MemoryBundle:
    short_term: str = ""
    long_term: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    updated: bool = False
    compressed: bool = False
