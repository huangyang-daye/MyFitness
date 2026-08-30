"""Agent 写库工具。

均用 LangChain `@tool` 修饰；`session` / `user_id` 通过 `InjectedToolArg` 注入。
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import BodyMetricRepository, NutritionLogRepository


@tool
def apply_body_manual_write(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    payload: dict,
) -> list[str]:
    """把用户手动录入的身体指标写入数据库。

    Args:
        payload: 形如 {"records": [{"record_date": "YYYY-MM-DD", "metric_type": "weight",
            "value": 70.5, "unit": "kg"}, ...]}。
    """
    repo = BodyMetricRepository(session, user_id)
    written: list[str] = []
    for r in payload.get("records", []):
        repo.upsert_manual(
            record_date=date.fromisoformat(r["record_date"]),
            metric_type=r["metric_type"],
            value=float(r["value"]),
            unit=r["unit"],
        )
        written.append(f"{r['record_date']} {r['metric_type']}={r['value']}{r['unit']}")
    return written


@tool
def apply_nutrition_manual_write(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    payload: dict,
) -> list[str]:
    """把用户手动录入的饮食记录写入数据库。

    Args:
        payload: 形如 {"items": [{"record_date": "YYYY-MM-DD", "meal_type": "午餐",
            "food_name": "鸡胸肉", "amount": 150, "unit": "g",
            "nutrients_snapshot": {"cal": 240, ...}}, ...]}。
    """
    repo = NutritionLogRepository(session, user_id)
    written: list[str] = []
    for item in payload.get("items", []):
        repo.add_manual(
            record_date=date.fromisoformat(item["record_date"]),
            meal_type=item["meal_type"],
            food_name=item["food_name"],
            amount=float(item["amount"]),
            unit=item["unit"],
            nutrients_snapshot=item["nutrients_snapshot"],
        )
        ntr = item["nutrients_snapshot"]
        written.append(
            f"{item['food_name']} {item['amount']}{item['unit']} (~{ntr.get('cal', 0)} kcal)"
        )
    return written
