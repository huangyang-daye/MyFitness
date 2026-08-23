"""Agent 结构化输出 — 对应 PRD §7.2–7.5。"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from myfitness.schemas.constants import DISCLAIMER


class CurrentMetrics(BaseModel):
    weight_kg: float | None = None
    bodyfat_pct: float | None = None
    measurements: dict[str, float] = Field(default_factory=dict)


class BodyTrend(BaseModel):
    period_days: int = 0
    weight_change_kg: float | None = None
    bodyfat_change_pct: float | None = None
    trend_direction: Literal["up", "down", "stable", "insufficient_data"] = "insufficient_data"
    weekly_avg_weight_kg: float | None = None


class BodyAnomaly(BaseModel):
    date: date
    metric: str
    description: str
    severity: Literal["info", "warning"] = "info"


class GoalProgress(BaseModel):
    goal_type: str | None = None
    target_value: float | None = None
    current_value: float | None = None
    progress_pct: float | None = None
    estimated_target_date: date | None = None
    milestones: list[dict] = Field(default_factory=list)


class BodyAgentOutput(BaseModel):
    agent: Literal["body_monitor"] = "body_monitor"
    analysis_date: date
    current_metrics: CurrentMetrics = Field(default_factory=CurrentMetrics)
    trend: BodyTrend = Field(default_factory=BodyTrend)
    anomalies: list[BodyAnomaly] = Field(default_factory=list)
    goal_progress: GoalProgress | None = None
    recommendations: list[str] = Field(default_factory=list)
    monitoring_frequency: Literal["daily", "every_other_day", "weekly"] = "daily"
    narrative: str = ""


class DailyTotals(BaseModel):
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class TdeeEstimate(BaseModel):
    method: Literal["mifflin_st_jeor"] = "mifflin_st_jeor"
    bmr: float | None = None
    activity_factor: float | None = None
    tdee: float | None = None
    target_calories: float | None = None
    goal_mode: Literal["cut", "bulk", "maintain"] | None = None


class NutritionBalance(BaseModel):
    calorie_delta: float | None = None
    protein_per_kg: float | None = None
    assessment: Literal["deficit", "surplus", "on_target", "unknown"] = "unknown"


class MealAnalysisItem(BaseModel):
    meal_type: str
    calories: float = 0
    comment: str = ""


class NutritionAgentOutput(BaseModel):
    agent: Literal["nutritionist"] = "nutritionist"
    analysis_date: date
    daily_totals: DailyTotals = Field(default_factory=DailyTotals)
    tdee_estimate: TdeeEstimate | None = None
    balance: NutritionBalance | None = None
    meal_analysis: list[MealAnalysisItem] = Field(default_factory=list)
    tomorrow_suggestions: dict | None = None
    recommendations: list[str] = Field(default_factory=list)
    narrative: str = ""


class RecentTrainingSummary(BaseModel):
    sessions_last_7d: int = 0
    total_volume_kg: float | None = None
    avg_rpe: float | None = None
    muscle_groups_trained: list[str] = Field(default_factory=list)


class MovementPlan(BaseModel):
    name: str
    sets: int = 0
    reps: str = ""
    weight: str = ""
    rest_seconds: int | None = None
    notes: str = ""


class TodayPlan(BaseModel):
    session_type: Literal["strength", "cardio", "rest", "active_recovery"] = "rest"
    focus: str = ""
    movements: list[MovementPlan] = Field(default_factory=list)
    estimated_duration_min: int | None = None


class WeeklyOutlineItem(BaseModel):
    date: date
    session_type: str = ""
    focus: str = ""


class ProgressiveOverloadItem(BaseModel):
    movement: str
    suggestion: str
    basis: str = ""


class FitnessAgentOutput(BaseModel):
    agent: Literal["fitness_planner"] = "fitness_planner"
    analysis_date: date
    recent_training_summary: RecentTrainingSummary = Field(default_factory=RecentTrainingSummary)
    recovery_assessment: Literal[
        "well_recovered", "moderate_fatigue", "high_fatigue", "unknown"
    ] = "unknown"
    today_plan: TodayPlan | None = None
    weekly_outline: list[WeeklyOutlineItem] = Field(default_factory=list)
    progressive_overload: list[ProgressiveOverloadItem] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    narrative: str = ""


class CrossDomainInsight(BaseModel):
    insight: str
    domains: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class ConflictResolved(BaseModel):
    conflict: str
    resolution: str


class ActionItem(BaseModel):
    priority: Literal["high", "medium", "low"] = "medium"
    action: str
    domain: str = ""


class SummaryAgentOutput(BaseModel):
    agent: Literal["summary"] = "summary"
    output_type: Literal["daily_report", "chat_reply"]
    content_md: str = ""
    cross_domain_insights: list[CrossDomainInsight] = Field(default_factory=list)
    conflicts_resolved: list[ConflictResolved] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class AgentOutputs(BaseModel):
    body: BodyAgentOutput | None = None
    nutrition: NutritionAgentOutput | None = None
    fitness: FitnessAgentOutput | None = None
    summary: SummaryAgentOutput | None = None
