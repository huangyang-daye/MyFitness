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
    short_term: str = ""  # 工作记忆（当前会话热窗口）
    episodic: str = ""  # 情景记忆（历史对话摘要）
    long_term: str = ""  # 用户画像
    profile: dict[str, Any] = field(default_factory=dict)
    updated: bool = False
    compressed: bool = False
    working_backend: str = "memory"  # redis | memory
