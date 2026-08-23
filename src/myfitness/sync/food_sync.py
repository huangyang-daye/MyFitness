from sqlalchemy.orm import Session

from myfitness.db.repositories.metrics import FoodRepository, NutritionLogRepository
from myfitness.xunji.client import XunjiClient
from myfitness.xunji.parsers.food import iter_food_entries


def sync_nutrition_logs(
    session: Session,
    client: XunjiClient,
    user_id: int,
    start_date,
    end_date,
) -> dict[str, int | list[str]]:
    food_repo = FoodRepository(session)
    log_repo = NutritionLogRepository(session, user_id)
    stats: dict[str, int | list[str]] = {"fetched": 0, "upserted": 0, "notes": []}

    result, notes = client.food.query_range_merged(
        start_date.isoformat(),
        end_date.isoformat(),
        include_detail=True,
    )
    stats["notes"] = notes

    for entry in iter_food_entries(result):
        stats["fetched"] += 1
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
