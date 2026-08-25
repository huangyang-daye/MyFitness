"""饮食数据响应解析 — skills/xunji-food-open-api/SKILL.md"""

from datetime import date, datetime
from typing import Any, Iterator

from myfitness.xunji.skills import MEAL_TYPES

MEAL_TYPE_ALIASES = {
    "morning": "breakfast",
    "breakfast": "breakfast",
    "noon": "lunch",
    "noon-added": "lunch",
    "lunch": "lunch",
    "night": "dinner",
    "dinner": "dinner",
    "snack": "snack",
}


def parse_record_date(value: str) -> date:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def normalize_meal_type(value: str | None) -> str:
    if not value:
        return "other"
    normalized = str(value).lower().strip()
    normalized = MEAL_TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in MEAL_TYPES else "other"


def calc_nutrients_from_ntr(
    ntr: dict | None,
    amount: float,
    gram_per_unit: float = 1.0,
) -> dict[str, float]:
    """Skill: ntr 为每 100g 的 cal/protein/fat/carb。"""
    ntr = ntr or {}
    factor = amount * gram_per_unit / 100.0
    return {
        "cal": round(float(ntr.get("cal", 0)) * factor, 2),
        "protein": round(float(ntr.get("protein", 0)) * factor, 2),
        "fat": round(float(ntr.get("fat", 0)) * factor, 2),
        "carb": round(float(ntr.get("carb", 0)) * factor, 2),
    }


def parse_search_foods(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Skill: 优先 res.foods；res.d 为压缩数组。"""
    foods = search_result.get("foods")
    if isinstance(foods, list):
        return foods

    compressed = search_result.get("d")
    if not isinstance(compressed, list):
        return []

    parsed: list[dict[str, Any]] = []
    for row in compressed:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            continue
        parsed.append(
            {
                "id": row[0],
                "name": row[1],
                "ntr": {
                    "cal": row[2],
                    "carb": row[3],
                    "fat": row[4],
                    "protein": row[5],
                },
                "foodpic": row[6] if len(row) > 6 else "",
                "uniquekey": row[7] if len(row) > 7 else None,
                "units": row[8] if len(row) > 8 else [],
            }
        )
    return parsed


def _flatten_day(day: dict[str, Any]) -> list[dict[str, Any]]:
    day_date = day.get("date") or day.get("datestr")
    entries: list[dict[str, Any]] = []

    for meal in day.get("meals") or []:
        if not isinstance(meal, dict):
            continue
        meal_type = meal.get("meal_type") or meal.get("type") or meal.get("name")
        for food in meal.get("foods") or meal.get("items") or []:
            if isinstance(food, dict):
                entries.append(
                    {
                        **food,
                        "date": food.get("date") or day_date,
                        "meal_type": food.get("meal_type") or meal_type,
                    }
                )

    foods = day.get("foods")
    if isinstance(foods, dict):
        for food in foods.get("records") or []:
            if isinstance(food, dict):
                entries.append({**food, "date": food.get("date") or day_date})
    else:
        for food in foods or []:
            if isinstance(food, dict):
                entries.append({**food, "date": food.get("date") or day_date})

    return entries


def _build_xunji_record_id(raw: dict[str, Any], datestr: str, meal_type: str, unit: str) -> str | None:
    food_ref = raw.get("id") or raw.get("record_id") or raw.get("uniquekey") or raw.get("name")
    if not food_ref:
        return None
    return ":".join(
        [
            "food",
            str(datestr)[:10],
            meal_type,
            str(food_ref),
            str(raw.get("uniquekey") or raw.get("name") or ""),
            unit,
        ]
    )


def iter_food_entries(query_result: dict[str, Any] | list) -> Iterator[dict[str, Any]]:
    """标准化饮食 query 结果为可入库条目。"""
    if isinstance(query_result, list):
        sources = query_result
    elif isinstance(query_result.get("days"), list):
        sources = []
        for day in query_result["days"]:
            sources.extend(_flatten_day(day))
    else:
        sources = []
        for key in ("foods", "records", "items"):
            value = query_result.get(key)
            if isinstance(value, list):
                sources = value
                break

    for raw in sources:
        datestr = raw.get("date") or raw.get("datestr")
        if not datestr:
            continue

        ntr = raw.get("ntr") or {}
        nutrients = raw.get("nutrients_snapshot") or raw.get("nutrients")
        amount = float(raw.get("amount") or raw.get("count") or 0)
        unit = str(raw.get("unit") or "g")
        meal_type = normalize_meal_type(raw.get("meal_type") or raw.get("meal"))

        if not nutrients and ntr:
            nutrients = calc_nutrients_from_ntr(ntr, amount)

        yield {
            "record_date": parse_record_date(str(datestr)),
            "meal_type": meal_type,
            "food_name": raw.get("name") or raw.get("food_name") or "unknown",
            "amount": amount,
            "unit": unit,
            "ntr": ntr,
            "nutrients_snapshot": nutrients or {"cal": 0, "protein": 0, "fat": 0, "carb": 0},
            "uniquekey": raw.get("uniquekey"),
            "units": raw.get("units"),
            "xunji_record_id": _build_xunji_record_id(raw, str(datestr), meal_type, unit),
        }


def format_food_write_summary(foods: list[dict[str, Any]]) -> str:
    lines = []
    for item in foods:
        lines.append(
            f"- {item.get('date')} {item.get('meal_type')}: "
            f"{item.get('name')} {item.get('amount')}{item.get('unit', '')}"
        )
    return "\n".join(lines)
