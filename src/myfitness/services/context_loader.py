"""从 DB 构建 Agent 上下文快照。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy.orm import Session

from myfitness.db.repositories.goals import UserGoalRepository
from myfitness.db.repositories.metrics import (
    BodyMetricRepository,
    NutritionLogRepository,
    TrainingLogRepository,
)
from myfitness.db.models import AgentPlan
from myfitness.schemas.state import ContextSnapshot, DateRange
from sqlalchemy import select


def load_context_snapshot(
    session: Session,
    user_id: int,
    end_date: date | None = None,
    lookback_days: int = 7,
    query_results: dict[str, dict] | None = None,
) -> ContextSnapshot:
    end = end_date or date.today()
    start = end - timedelta(days=lookback_days - 1)

    body_repo = BodyMetricRepository(session, user_id)
    nutrition_repo = NutritionLogRepository(session, user_id)
    training_repo = TrainingLogRepository(session, user_id)
    goal_repo = UserGoalRepository(session, user_id)

    qr = query_results or {}
    if "body" in qr:
        body_summary = _summarize_body_from_query(qr["body"])
    else:
        body_records = body_repo.query_range(start, end)
        body_summary = _summarize_body(body_repo, body_records, start, end)

    if "nutrition" in qr:
        nutrition_summary = _summarize_nutrition_from_query(qr["nutrition"], nutrition_repo, end)
    else:
        nutrition_records = nutrition_repo.query_range(start, end)
        nutrition_summary = _summarize_nutrition(nutrition_repo, nutrition_records, start, end)

    if "training" in qr:
        training_summary = _summarize_training_from_query(qr["training"])
    else:
        training_records = training_repo.query_range(start, end)
        training_summary = _summarize_training(training_records, start, end)

    goals = [
        {
            "goal_type": g.goal_type,
            "target_value": float(g.target_value),
            "start_value": float(g.start_value) if g.start_value is not None else None,
            "start_date": g.start_date.isoformat(),
            "target_date": g.target_date.isoformat() if g.target_date else None,
            "status": g.status,
        }
        for g in goal_repo.list_active()
    ]

    plans = session.scalars(
        select(AgentPlan).where(
            AgentPlan.user_id == user_id,
            AgentPlan.status == "active",
            AgentPlan.end_date >= start,
        )
    ).all()
    active_plans = [
        {
            "plan_type": p.plan_type,
            "start_date": p.start_date.isoformat(),
            "end_date": p.end_date.isoformat(),
            "plan_json": p.plan_json,
        }
        for p in plans
    ]

    data_gaps: list[str] = []
    if not body_summary.get("latest_weight_kg"):
        data_gaps.append("近7天缺少体重记录")
    if nutrition_summary.get("days_with_logs", 0) == 0:
        data_gaps.append("近7天缺少饮食记录")
    if training_summary.get("sessions", 0) == 0:
        data_gaps.append("近7天缺少训练记录")

    return ContextSnapshot(
        date_range=DateRange(start=start, end=end),
        body_metrics_summary=body_summary,
        nutrition_summary=nutrition_summary,
        training_summary=training_summary,
        user_goals=goals,
        active_plans=active_plans,
        data_gaps=data_gaps,
    )


def _summarize_body(
    repo: BodyMetricRepository,
    records: list,
    start: date,
    end: date,
) -> dict:
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        by_date[r.record_date.isoformat()][r.metric_type] = float(r.value)

    weights: list[tuple[date, float]] = []
    bodyfats: list[tuple[date, float]] = []
    for r in records:
        if r.metric_type == "weight":
            weights.append((r.record_date, float(r.value)))
        elif r.metric_type == "bodyfat":
            bodyfats.append((r.record_date, float(r.value)))

    weights.sort(key=lambda x: x[0])
    bodyfats.sort(key=lambda x: x[0])

    latest_weight = weights[-1][1] if weights else None
    latest_bodyfat = bodyfats[-1][1] if bodyfats else None
    weight_change = (weights[-1][1] - weights[0][1]) if len(weights) >= 2 else None

    return {
        "record_count": len(records),
        "days_with_data": len(by_date),
        "latest_weight_kg": latest_weight,
        "latest_bodyfat_pct": latest_bodyfat,
        "weight_change_kg": weight_change,
        "by_date": dict(by_date),
    }


def _summarize_body_from_query(data: dict) -> dict:
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    weights: list[tuple[str, float]] = []
    bodyfats: list[tuple[str, float]] = []
    for r in data.get("records", []):
        d = r["date"]
        by_date[d][r["metric_type"]] = float(r["value"])
        if r["metric_type"] == "weight":
            weights.append((d, float(r["value"])))
        elif r["metric_type"] == "bodyfat":
            bodyfats.append((d, float(r["value"])))
    weights.sort(key=lambda x: x[0])
    bodyfats.sort(key=lambda x: x[0])
    return {
        "record_count": data.get("count", len(data.get("records", []))),
        "days_with_data": len(by_date),
        "latest_weight_kg": weights[-1][1] if weights else None,
        "latest_bodyfat_pct": bodyfats[-1][1] if bodyfats else None,
        "weight_change_kg": (weights[-1][1] - weights[0][1]) if len(weights) >= 2 else None,
        "by_date": dict(by_date),
    }


def _summarize_nutrition_from_query(
    data: dict,
    repo: NutritionLogRepository,
    end: date,
) -> dict:
    _ = repo  # 保留签名，便于日后 fallback
    by_date = {
        d: {
            "calories": float(v.get("calories", 0)),
            "protein_g": float(v.get("protein_g", 0)),
            "carbs_g": float(v.get("carbs_g", 0)),
            "fat_g": float(v.get("fat_g", 0)),
        }
        for d, v in (data.get("daily_totals") or {}).items()
    }
    today = end.isoformat()
    empty = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    today_totals = by_date.get(today, empty)
    return {
        "record_count": data.get("count", 0),
        "days_with_logs": len(by_date),
        "today_totals": today_totals,
        "by_date": by_date,
    }


def _summarize_training_from_query(data: dict) -> dict:
    movements: set[str] = set()
    dates: list[str] = []
    total_volume = 0.0
    for s in data.get("sessions", []):
        dates.append(s["date"])
        if s.get("total_volume_kg"):
            total_volume += float(s["total_volume_kg"])
        for m in s.get("movements") or []:
            name = m.get("name") if isinstance(m, dict) else None
            if name:
                movements.add(name)
    return {
        "sessions": data.get("count", len(data.get("sessions", []))),
        "movements": sorted(movements),
        "dates": sorted(set(dates)),
        "total_volume_kg": round(total_volume, 1),
    }


def _summarize_nutrition(
    repo: NutritionLogRepository,
    records: list,
    start: date,
    end: date,
) -> dict:
    by_date: dict[str, dict[str, float]] = {}
    days_with_logs: set[str] = set()
    for r in records:
        d = r.record_date.isoformat()
        days_with_logs.add(d)
        if d not in by_date:
            by_date[d] = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        ntr = r.nutrients_snapshot or {}
        by_date[d]["calories"] += float(ntr.get("cal", 0) or 0)
        by_date[d]["protein_g"] += float(ntr.get("protein", 0) or 0)
        by_date[d]["carbs_g"] += float(ntr.get("carb", 0) or 0)
        by_date[d]["fat_g"] += float(ntr.get("fat", 0) or 0)

    today = end.isoformat()
    today_totals = by_date.get(today, repo.daily_totals(end))

    return {
        "record_count": len(records),
        "days_with_logs": len(days_with_logs),
        "today_totals": today_totals,
        "by_date": by_date,
    }


def _summarize_training(records: list, start: date, end: date) -> dict:
    from myfitness.xunji.parsers.training import parse_training_payload

    sessions = len(records)
    movements: set[str] = set()
    total_volume = 0.0
    for log in records:
        payload = log.raw_payload if isinstance(log.raw_payload, dict) else {}
        if payload.get("movements"):
            parsed = parse_training_payload(payload)
            total_volume += parsed.get("total_volume_kg") or 0
            for m in parsed.get("movements") or []:
                movements.add(m["name"])
        else:
            for ex in log.exercises:
                movements.add(ex.movement_name)

    return {
        "sessions": sessions,
        "movements": sorted(movements),
        "dates": sorted({log.record_date.isoformat() for log in records}),
        "total_volume_kg": round(total_volume, 1),
    }
