"""数据库查询 Tool — 按日期范围读取 body / nutrition / training 明细。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import (
    BodyMetricRepository,
    NutritionLogRepository,
    TrainingLogRepository,
)
from myfitness.xunji.parsers.training import parse_training_payload


def query_body_metrics(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    metric_type: str | None = None,
) -> dict:
    repo = BodyMetricRepository(session, user_id)
    records = repo.query_range(start_date, end_date, metric_type)
    return {
        "tool": "query_body_metrics",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "metric_type": metric_type,
        "count": len(records),
        "records": [
            {
                "date": r.record_date.isoformat(),
                "metric_type": r.metric_type,
                "value": float(r.value),
                "unit": r.unit,
                "source": r.source,
            }
            for r in records
        ],
    }


def query_nutrition_logs(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    meal_type: str | None = None,
) -> dict:
    repo = NutritionLogRepository(session, user_id)
    records = repo.query_range(start_date, end_date)
    if meal_type:
        records = [r for r in records if r.meal_type == meal_type]

    daily_totals: dict[str, dict[str, float]] = {}
    entries: list[dict] = []
    for r in records:
        d = r.record_date.isoformat()
        ntr = r.nutrients_snapshot or {}
        if d not in daily_totals:
            daily_totals[d] = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        daily_totals[d]["calories"] += float(ntr.get("cal", 0) or 0)
        daily_totals[d]["protein_g"] += float(ntr.get("protein", 0) or 0)
        daily_totals[d]["carbs_g"] += float(ntr.get("carb", 0) or 0)
        daily_totals[d]["fat_g"] += float(ntr.get("fat", 0) or 0)
        entries.append(
            {
                "date": d,
                "meal_type": r.meal_type,
                "food_name": r.food_name,
                "amount": float(r.amount),
                "unit": r.unit,
                "nutrients": {
                    "calories": float(ntr.get("cal", 0) or 0),
                    "protein_g": float(ntr.get("protein", 0) or 0),
                    "carbs_g": float(ntr.get("carb", 0) or 0),
                    "fat_g": float(ntr.get("fat", 0) or 0),
                },
                "source": r.source,
            }
        )

    return {
        "tool": "query_nutrition_logs",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "meal_type": meal_type,
        "count": len(entries),
        "daily_totals": daily_totals,
        "entries": entries,
    }


def query_training_logs(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
) -> dict:
    repo = TrainingLogRepository(session, user_id)
    records = repo.query_range(start_date, end_date)
    sessions: list[dict] = []
    for log in records:
        payload = log.raw_payload if isinstance(log.raw_payload, dict) else {}
        if payload.get("movements"):
            parsed = parse_training_payload(payload)
            sessions.append(
                {
                    "date": log.record_date.isoformat(),
                    "title": parsed.get("title") or log.title,
                    "source": log.source,
                    "localid": parsed.get("localid") or log.xunji_localid,
                    "duration_minutes": parsed.get("duration_minutes"),
                    "calories": parsed.get("calories"),
                    "total_sets": parsed.get("total_sets"),
                    "total_volume_kg": parsed.get("total_volume_kg"),
                    "movements": parsed.get("movements") or [],
                }
            )
        else:
            sessions.append(
                {
                    "date": log.record_date.isoformat(),
                    "title": log.title,
                    "source": log.source,
                    "localid": log.xunji_localid,
                    "movements": [
                        {
                            "name": ex.movement_name,
                            "set_count": ex.set_count,
                            "sets": ex.sets_detail or [],
                        }
                        for ex in log.exercises
                    ],
                }
            )

    return {
        "tool": "query_training_logs",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "count": len(sessions),
        "sessions": sessions,
    }


def execute_query_plan(
    session: Session,
    user_id: int,
    domains: list[str],
    start_date: date,
    end_date: date,
    metric_type: str | None = None,
    meal_type: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """按查询计划执行一个或多个 domain 的 DB 查询。"""
    from myfitness.graph.progress import emit, label_for

    results: dict[str, dict] = {}
    if "body" in domains:
        emit(on_progress, f"{label_for('query_body_metrics')}…")
        results["body"] = query_body_metrics(
            session, user_id, start_date, end_date, metric_type=metric_type
        )
    if "nutrition" in domains:
        emit(on_progress, f"{label_for('query_nutrition_logs')}…")
        results["nutrition"] = query_nutrition_logs(
            session, user_id, start_date, end_date, meal_type=meal_type
        )
    if "training" in domains or "fitness" in domains:
        emit(on_progress, f"{label_for('query_training_logs')}…")
        results["training"] = query_training_logs(session, user_id, start_date, end_date)
    return results
