"""任务计划与执行结果 — Planner / Orchestrator / Judge 共享契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ContextSnapshot, Intent


@dataclass
class PlannedTask:
    """Planner 拆分的单个子任务。"""

    id: str
    intent: Intent
    description: str
    domain: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent.value,
            "description": self.description,
            "domain": self.domain,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "depends_on": list(self.depends_on),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PlannedTask:
        start_raw = raw.get("start_date")
        end_raw = raw.get("end_date")
        return cls(
            id=str(raw["id"]),
            intent=Intent(str(raw["intent"])),
            description=str(raw.get("description") or ""),
            domain=raw.get("domain"),
            start_date=date.fromisoformat(start_raw) if start_raw else None,
            end_date=date.fromisoformat(end_raw) if end_raw else None,
            depends_on=[str(item) for item in raw.get("depends_on") or []],
            params=dict(raw.get("params") or {}),
        )


@dataclass
class TaskPlan:
    """一轮用户消息对应的完整任务计划。"""

    tasks: list[PlannedTask]
    user_requirements: str = ""
    primary_intent: Intent = Intent.GENERAL
    domain: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "user_requirements": self.user_requirements,
            "primary_intent": self.primary_intent.value,
            "domain": self.domain,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskPlan:
        start_raw = raw.get("start_date")
        end_raw = raw.get("end_date")
        return cls(
            tasks=[PlannedTask.from_dict(item) for item in raw.get("tasks") or []],
            user_requirements=str(raw.get("user_requirements") or ""),
            primary_intent=Intent(str(raw.get("primary_intent") or Intent.GENERAL.value)),
            domain=raw.get("domain"),
            start_date=date.fromisoformat(start_raw) if start_raw else None,
            end_date=date.fromisoformat(end_raw) if end_raw else None,
        )


@dataclass
class TaskResult:
    task_id: str
    intent: Intent
    status: str  # success | failed | skipped | pending_confirmation
    summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "intent": self.intent.value,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class ExecutionResult:
    task_results: list[TaskResult] = field(default_factory=list)
    agent_outputs: AgentOutputs = field(default_factory=AgentOutputs)
    context: ContextSnapshot | None = None
    tools_invoked: list[str] = field(default_factory=list)
    agents_invoked: list[str] = field(default_factory=list)
    reply_parts: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    errors: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [part for part in self.reply_parts if part.strip()]
        for result in self.task_results:
            if result.summary:
                lines.append(result.summary)
            if result.error:
                lines.append(result.error)
        return "\n\n".join(lines)


@dataclass
class JudgeVerdict:
    satisfied: bool
    feedback: str = ""
    missing: list[str] = field(default_factory=list)
    retry_task_ids: list[str] = field(default_factory=list)
