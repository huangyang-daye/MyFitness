from datetime import date

from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import FoodRepository, NutritionLogRepository
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.food import clamp_food_query_range
from myfitness.xunji.parsers.food import iter_food_entries


def sync_nutrition_logs(
    session: Session,
    client: XunjiClient,
    user_id: int,
    start_date: date,
    end_date: date,
) -> dict[str, int | list[str]]:
    food_repo = FoodRepository(session)
    log_repo = NutritionLogRepository(session, user_id)
    stats: dict[str, int | list[str]] = {"fetched": 0, "upserted": 0, "notes": []}

    clamped_start, clamped_end, notes = clamp_food_query_range(start_date, end_date)
    stats["notes"] = notes

    query_chunks = client.food.query_range(
        clamped_start.isoformat(),
        clamped_end.isoformat(),
    )

    entries = []
    for chunk in query_chunks:
        entries.extend(iter_food_entries(chunk))

    stats["fetched"] = len(entries)

    for entry in entries:
        food = food_repo.get_or_create(
            name=entry["food_name"],
            uniquekey=entry.get("uniquekey"),
            ntr=entry.get("ntr") or {"cal": 0, "protein": 0, "fat": 0, "carb": 0},
            units=entry.get("units"),
        )

        log_repo.upsert_from_sync(
            record_date=entry["record_date"],
            meal_type=entry["meal_type"],
            food_name=entry["food_name"],
            amount=entry["amount"],
            unit=entry["unit"],
            nutrients_snapshot=entry["nutrients_snapshot"],
            food_id=food.id,
            xunji_record_id=entry.get("xunji_record_id"),
        )
        stats["upserted"] += 1

    return stats
