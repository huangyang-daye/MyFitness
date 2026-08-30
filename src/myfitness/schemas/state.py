"""LangGraph State 与对话契约 — 对应 PRD §7.1。"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from myfitness.schemas.agent_outputs import AgentOutputs


class RunMode(StrEnum):
    DAILY_REPORT = "daily_report"
    CHAT = "chat"


class Intent(StrEnum):
    DATA_QUERY = "data_query"
    MANUAL_ENTRY = "manual_entry"
    PLAN_ADJUST = "plan_adjust"
    TREND_ANALYSIS = "trend_analysis"
    GOAL_SETTING = "goal_setting"
    SYNC_TRIGGER = "sync_trigger"
    SCHEDULE_MANAGE = "schedule_manage"
    REPORT_TRIGGER = "report_trigger"
    CHART_TRIGGER = "chart_trigger"
    GENERAL = "general"
    CONFIRMATION_RESPONSE = "confirmation_response"


class RouteResult:
    """意图识别结果 — 支持一条消息多个意图（按执行顺序）+ 显式日期范围。

    - intents：非空有序意图列表；「同步X并生成日报」→ [SYNC_TRIGGER, REPORT_TRIGGER]
    - start_date/end_date：消息中明确指出的日期范围（用于同步/日报），未指明时为 None
    - intent：主意图（intents[0]），保持向后兼容
    """

    def __init__(
        self,
        intents: Intent | list[Intent] | None = None,
        domain: str | None = None,
        confirmation_action: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> None:
        if intents is None:
            items: list[Intent] = []
        elif isinstance(intents, Intent):
            items = [intents]
        else:
            items = []
            for item in intents:
                if isinstance(item, Intent):
                    items.append(item)
                elif isinstance(item, str):
                    try:
                        items.append(Intent(item))
                    except ValueError:
                        continue
        # 去重且保持顺序
        self.intents: list[Intent] = list(dict.fromkeys(items))
        self.domain = domain  # body | nutrition | fitness
        self.confirmation_action = confirmation_action  # confirm | cancel
        self.start_date = start_date
        self.end_date = end_date

    @property
    def intent(self) -> Intent:
        return self.intents[0] if self.intents else Intent.GENERAL

    def has(self, intent: Intent) -> bool:
        return intent in self.intents

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"RouteResult(intents={[i.value for i in self.intents]}, domain={self.domain!r}, "
            f"start={self.start_date}, end={self.end_date})"
        )


class Artifact(BaseModel):
    """会话产物 — 对话过程中生成并落盘的文件（报告、统计图文档等）。

    path 是绝对路径；读取前必须校验落在 data_dir 之内（见 services/artifacts.py）。
    """

    id: str
    kind: str = "report"  # report | chart
    title: str = ""
    subtitle: str = ""  # 日期 / 周期 / 指标说明
    path: str = ""
    created_at: datetime | None = None


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime | None = None
    artifacts: list[Artifact] = Field(default_factory=list)


class DateRange(BaseModel):
    start: date
    end: date


class ContextSnapshot(BaseModel):
    date_range: DateRange
    body_metrics_summary: dict[str, Any] = Field(default_factory=dict)
    nutrition_summary: dict[str, Any] = Field(default_factory=dict)
    training_summary: dict[str, Any] = Field(default_factory=dict)
    user_goals: list[dict[str, Any]] = Field(default_factory=list)
    active_plans: list[dict[str, Any]] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    query_results: dict[str, Any] = Field(default_factory=dict)


class PendingConfirmation(BaseModel):
    # db_write | xunji_write | plan_update | schedule_upsert | schedule_cancel | *_clarification
    action_type: str
    summary: str
    payload: dict[str, Any]
    expires_at: datetime
    domain: str | None = None  # body | nutrition | schedule


class GraphMetadata(BaseModel):
    started_at: datetime | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    agents_invoked: list[str] = Field(default_factory=list)
    tools_invoked: list[str] = Field(default_factory=list)


class MyFitnessGraphState(BaseModel):
    user_id: int = 1
    session_id: str = "default"
    mode: RunMode = RunMode.CHAT
    target_date: date | None = None
    user_message: str = ""
    intent: Intent | None = None
    intent_domain: str | None = None  # body | nutrition | fitness
    messages: list[ChatMessage] = Field(default_factory=list)
    context: ContextSnapshot | None = None
    agent_outputs: AgentOutputs = Field(default_factory=AgentOutputs)
    pending_confirmation: PendingConfirmation | None = None
    errors: list[str] = Field(default_factory=list)
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)
    reply: str = ""
    # 本轮产生的产物缓冲区：_append_assistant 挂到消息上后清空
    pending_artifacts: list[Artifact] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


def merge_errors(existing: list[str], new: list[str]) -> list[str]:
    return existing + [e for e in new if e not in existing]
