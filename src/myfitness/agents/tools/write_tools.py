"""Agent 写库工具。"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import BodyMetricRepository, NutritionLogRepository


def apply_body_manual_write(session: Session, user_id: int, payload: dict) -> list[str]:
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


def apply_nutrition_manual_write(session: Session, user_id: int, payload: dict) -> list[str]:
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
