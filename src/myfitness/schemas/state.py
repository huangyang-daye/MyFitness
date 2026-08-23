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
    GENERAL = "general"
    CONFIRMATION_RESPONSE = "confirmation_response"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime | None = None


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
    action_type: str  # db_write | xunji_write | plan_update
    summary: str
    payload: dict[str, Any]
    expires_at: datetime
    domain: str | None = None  # body | nutrition


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

    model_config = {"extra": "ignore"}


def merge_errors(existing: list[str], new: list[str]) -> list[str]:
    return existing + [e for e in new if e not in existing]
