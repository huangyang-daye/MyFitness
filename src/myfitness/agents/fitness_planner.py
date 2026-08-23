"""FitnessPlannerAgent — 训练计划分析。"""

from __future__ import annotations

from datetime import date, timedelta

from myfitness.schemas.agent_outputs import (
    FitnessAgentOutput,
    RecentTrainingSummary,
    TodayPlan,
    WeeklyOutlineItem,
)
from myfitness.schemas.state import ContextSnapshot
from myfitness.xunji.parsers.training import format_training_session


def run_fitness_agent(
    context: ContextSnapshot, analysis_date: date | None = None
) -> FitnessAgentOutput:
    analysis_date = analysis_date or context.date_range.end
    summary = context.training_summary

    recent = RecentTrainingSummary(
        sessions_last_7d=int(summary.get("sessions", 0)),
        muscle_groups_trained=summary.get("movements", [])[:10],
    )

    if training_query := (context.query_results or {}).get("training"):
        recent.sessions_last_7d = training_query.get("count", recent.sessions_last_7d)
        names: list[str] = []
        for s in training_query.get("sessions", []):
            for m in s.get("movements", []):
                name = m.get("name") or m.get("movement_name")
                if name:
                    names.append(name)
        if names:
            recent.muscle_groups_trained = names[:10]

    recovery = "unknown"
    if recent.sessions_last_7d >= 5:
        recovery = "moderate_fatigue"
    elif recent.sessions_last_7d >= 1:
        recovery = "well_recovered"
    elif recent.sessions_last_7d == 0:
        recovery = "well_recovered"

    today_plan = TodayPlan(
        session_type="rest" if recent.sessions_last_7d >= 5 else "strength",
        focus="休息日" if recent.sessions_last_7d >= 5 else "按计划进行力量训练",
    )

    weekly: list[WeeklyOutlineItem] = []
    for i in range(7):
        d = analysis_date - timedelta(days=6 - i)
        weekly.append(
            WeeklyOutlineItem(
                date=d,
                session_type="strength" if d.weekday() in {0, 2, 4} else "rest",
                focus="推/拉/腿分化" if d.weekday() in {0, 2, 4} else "恢复",
            )
        )

    narrative = _build_narrative(context, recent)

    return FitnessAgentOutput(
        analysis_date=analysis_date,
        recent_training_summary=recent,
        recovery_assessment=recovery,
        today_plan=today_plan,
        weekly_outline=weekly,
        narrative=narrative,
    )


def _build_narrative(context: ContextSnapshot, recent: RecentTrainingSummary) -> str:
    training_query = (context.query_results or {}).get("training")
    if training_query and training_query.get("sessions"):
        lines = [f"查询到 {training_query['count']} 次训练记录："]
        for s in training_query["sessions"][:3]:
            movements = s.get("movements") or []
            if movements and isinstance(movements[0].get("sets"), list):
                parsed = {
                    "date": s["date"],
                    "title": s.get("title"),
                    "movement_count": len(s.get("movements") or []),
                    "duration_minutes": s.get("duration_minutes"),
                    "calories": s.get("calories"),
                    "total_volume_kg": s.get("total_volume_kg"),
                    "movements": s.get("movements") or [],
                }
                lines.append(format_training_session(parsed))
            else:
                names = ", ".join(
                    m.get("name") or m.get("movement_name", "") for m in s.get("movements", [])
                )
                lines.append(f"- {s['date']} {s.get('title') or '训练'}: {names}")
        return "\n".join(lines)

    if recent.sessions_last_7d:
        return f"近 7 天完成 {recent.sessions_last_7d} 次训练。"
    return "近 7 天暂无训练记录。"
