"""数据库查询 Tool — 按日期范围读取 body / nutrition / training 明细。

全部函数均用 LangChain `@tool` 修饰（详见 `base.py` 的调用约定）。
`session` / `user_id` 通过 `InjectedToolArg` 注入，不会进入 LLM 入参模式。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from myfitness.agents.tools.base import invoke_tool
from myfitness.db.repositories.metrics import (
    BodyMetricRepository,
    NutritionLogRepository,
    TrainingLogRepository,
)
from myfitness.xunji.parsers.training import parse_training_payload


def _collect_latest_body_metrics(repo: BodyMetricRepository) -> dict:
    latest: dict[str, dict] = {}
    for metric_type in ("weight", "bodyfat"):
        row = repo.get_latest(metric_type)
        if row is None:
            continue
        latest[metric_type] = {
            "value": float(row.value),
            "unit": row.unit,
            "date": row.record_date.isoformat(),
            "source": row.source,
        }
    return latest


@tool
def query_body_metrics(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    metric_type: str | None = None,
) -> dict:
    """查询指定日期范围内用户的身体指标记录（体重 / 体脂 / 各围度等）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        metric_type: 指标类型，如 weight / bodyfat / weist；为空返回全部类型。
    """
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
        "latest_metrics": _collect_latest_metrics(repo),
    }


def _collect_latest_metrics(repo: BodyMetricRepository) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for metric_type in ("weight", "bodyfat"):
        row = repo.get_latest(metric_type)
        if row is None:
            continue
        latest[metric_type] = {
            "value": float(row.value),
            "unit": row.unit,
            "date": row.record_date.isoformat(),
            "source": row.source,
        }
    return latest


@tool
def query_nutrition_logs(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    meal_type: str | None = None,
) -> dict:
    """查询指定日期范围内的饮食记录，并按日汇总热量 / 蛋白 / 碳水 / 脂肪。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        meal_type: 餐型过滤，如 早餐 / 午餐 / 晚餐 / 加餐；为空返回全部。
    """
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


@tool
def query_training_logs(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
) -> dict:
    """查询指定日期范围内的训练记录（动作、组数、容量、消耗等）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
    """
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
                    # raw_payload 缺 title 时回退 DB 列，避免统一显示「训练」
                    "title": payload.get("title") or log.title or parsed.get("title"),
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


@tool
def execute_query_plan(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    domains: list[str],
    start_date: date,
    end_date: date,
    metric_type: str | None = None,
    meal_type: str | None = None,
    include_latest_body: bool = False,
    muscle_group: str | None = None,
    on_progress: Annotated[Callable | None, InjectedToolArg] = None,
) -> dict[str, dict]:
    """按查询计划一次性执行一个或多个 domain（body / nutrition / training）的 DB 查询。

    Args:
        domains: 要查询的域列表，如 ["body", "training"]。
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        metric_type: 身体指标过滤，透传给 body 查询。
        meal_type: 餐型过滤，透传给 nutrition 查询。
    """
    from myfitness.graph.progress import emit, label_for
    from myfitness.db.sql_logging import log_query_context

    results: dict[str, dict] = {}
    log_query_context(
        "execute_query_plan",
        domains=domains,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        metric_type=metric_type,
        meal_type=meal_type,
        include_latest_body=include_latest_body,
        muscle_group=muscle_group,
    )
    if "body" in domains:
        emit(on_progress, f"{label_for('query_body_metrics')}…")
        results["body"] = invoke_tool(
            query_body_metrics,
            session,
            user_id,
            start_date=start_date,
            end_date=end_date,
            metric_type=metric_type,
        )
        if include_latest_body:
            results["body"]["latest_metrics"] = _collect_latest_body_metrics(
                BodyMetricRepository(session, user_id)
            )
    if "nutrition" in domains:
        emit(on_progress, f"{label_for('query_nutrition_logs')}…")
        results["nutrition"] = invoke_tool(
            query_nutrition_logs,
            session,
            user_id,
            start_date=start_date,
            end_date=end_date,
            meal_type=meal_type,
        )
    if "training" in domains or "fitness" in domains:
        emit(on_progress, f"{label_for('query_training_logs')}…")
        training = invoke_tool(
            query_training_logs, session, user_id, start_date=start_date, end_date=end_date
        )
        if muscle_group:
            training = _filter_training_by_muscle(training, muscle_group)
        results["training"] = training
    return results


def _filter_training_by_muscle(data: dict, muscle_group: str) -> dict:
    sessions: list[dict] = []
    for session in data.get("sessions", []):
        movements = []
        for movement in session.get("movements", []):
            muscle_type = str(movement.get("muscle_type") or movement.get("type") or "")
            if muscle_group in muscle_type:
                movements.append(movement)
        if movements:
            sessions.append({**session, "movements": movements})
    return {
        **data,
        "sessions": sessions,
        "count": len(sessions),
        "muscle_group": muscle_group,
    }
