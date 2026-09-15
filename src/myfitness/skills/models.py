"""Skill 规格与运行时上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from myfitness.graph.progress import ProgressCallback
from myfitness.schemas.state import Intent


@dataclass(frozen=True)
class SkillSpec:
    """从 SKILL.md 解析出的 Skill 元数据。"""

    name: str
    description: str
    path: Path
    source: str  # builtin | project
    handler: str | None = None
    when: str = "context"  # context | task | both
    enabled: bool = True
    triggers: dict[str, list[str]] = field(default_factory=dict)
    body: str = ""

    @property
    def has_handler(self) -> bool:
        return bool(self.handler)

    @property
    def runs_in_context(self) -> bool:
        return self.when in {"context", "both"}

    @property
    def runs_as_task(self) -> bool:
        return self.when in {"task", "both"}


@dataclass
class SkillContext:
    """Skill handler 的输入。"""

    session: Session
    user_id: int
    message: str
    intent: Intent
    domain: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    today: date | None = None
    plan: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    on_progress: ProgressCallback | None = None


@dataclass
class SkillResult:
    """Skill handler 的输出。"""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    tools_invoked: list[str] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
