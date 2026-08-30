"""BodyMonitorAgent — 基于上下文的数据分析。"""

from __future__ import annotations

from datetime import date

from myfitness.debug import trace_agent
from myfitness.schemas.agent_outputs import (
    BodyAgentOutput,
    BodyTrend,
    CurrentMetrics,
    GoalProgress,
)
from myfitness.schemas.state import ContextSnapshot


@trace_agent("BodyMonitorAgent")
def run_body_agent(context: ContextSnapshot, analysis_date: date | None = None) -> BodyAgentOutput:
    analysis_date = analysis_date or context.date_range.end
    summary = context.body_metrics_summary
    current = _current_metrics_from_context(context, summary)

    change = summary.get("weight_change_kg")
    direction = "insufficient_data"
    if change is not None:
        if change > 0.3:
            direction = "up"
        elif change < -0.3:
            direction = "down"
        else:
            direction = "stable"

    trend = BodyTrend(
        period_days=(context.date_range.end - context.date_range.start).days + 1,
        weight_change_kg=change,
        trend_direction=direction,
        weekly_avg_weight_kg=summary.get("latest_weight_kg"),
    )

    goal_progress = None
    for g in context.user_goals:
        if g.get("goal_type") == "weight" and current.weight_kg is not None:
            target = g.get("target_value")
            start_val = g.get("start_value") or current.weight_kg
            if target:
                total = abs(target - start_val) or 1
                done = abs(start_val - current.weight_kg)
                goal_progress = GoalProgress(
                    goal_type="weight",
                    target_value=target,
                    current_value=current.weight_kg,
                    progress_pct=round(min(100, done / total * 100), 1),
                )

    recommendations: list[str] = []
    if direction == "up":
        recommendations.append("近期体重呈上升趋势，建议结合饮食与训练数据综合评估。")
    elif direction == "down":
        recommendations.append("近期体重呈下降趋势，请确保蛋白质摄入与恢复充足。")

    narrative = _build_narrative(current, trend, goal_progress)

    return BodyAgentOutput(
        analysis_date=analysis_date,
        current_metrics=current,
        trend=trend,
        goal_progress=goal_progress,
        recommendations=recommendations,
        narrative=narrative,
    )


def _current_metrics_from_context(context: ContextSnapshot, summary: dict) -> CurrentMetrics:
    body_query = (context.query_results or {}).get("body")
    if body_query and body_query.get("records"):
        latest_weight = None
        latest_bodyfat = None
        # records 按 record_date desc 排序，取每种指标的第一条即为最新
        for r in body_query["records"]:
            if r["metric_type"] == "weight" and latest_weight is None:
                latest_weight = r["value"]
            elif r["metric_type"] == "bodyfat" and latest_bodyfat is None:
                latest_bodyfat = r["value"]
        return CurrentMetrics(weight_kg=latest_weight, bodyfat_pct=latest_bodyfat)

    return CurrentMetrics(
        weight_kg=summary.get("latest_weight_kg"),
        bodyfat_pct=summary.get("latest_bodyfat_pct"),
    )


def _build_narrative(current: CurrentMetrics, trend: BodyTrend, goal: GoalProgress | None) -> str:
    parts: list[str] = []
    if current.weight_kg:
        parts.append(f"最新体重 {current.weight_kg} kg。")
    if current.bodyfat_pct:
        parts.append(f"最新体脂 {current.bodyfat_pct}%。")
    if trend.weight_change_kg is not None:
        parts.append(f"观察期内体重变化 {trend.weight_change_kg:+.1f} kg。")
    if goal and goal.progress_pct is not None:
        parts.append(f"体重目标完成度约 {goal.progress_pct}%。")
    return " ".join(parts) or "暂无足够身体数据进行分析。"
