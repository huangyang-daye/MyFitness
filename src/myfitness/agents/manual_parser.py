"""手动录入解析 — 体重/饮食。"""

from __future__ import annotations

import re
from datetime import date

from myfitness.agents.tools.query_planner import parse_single_date

MEAL_KEYWORDS = {
    "breakfast": ["早餐", "早饭", "早上"],
    "lunch": ["午餐", "午饭", "中午"],
    "dinner": ["晚餐", "晚饭", "晚上"],
    "snack": ["零食", "加餐"],
}

_WEIGHT_RE = re.compile(
    r"(?:初始?\s*)?(?:体重|weight)\s*(?:为|是|:|：)?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
    re.I,
)
_BODYFAT_RE = re.compile(
    r"(?:初始?\s*)?(?:体脂(?:率)?|bodyfat)\s*(?:为|是|:|：)?\s*(\d+(?:\.\d+)?)\s*%?",
    re.I,
)
_WEIGHT_FALLBACK_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)",
    re.I,
)
_GOAL_WEIGHT_RE = re.compile(
    r"(?:目标|减到|降到|增到|达到)\s*(?:体重)?\s*(\d+(?:\.\d+)?)\s*(?:kg|公斤|千克)?",
    re.I,
)


_YEAR_CN_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def _extract_record_date(message: str) -> date | None:
    match = _YEAR_CN_DATE_RE.search(message)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    try:
        return parse_single_date(message)
    except ValueError:
        return None


def parse_body_entry(message: str, target_date: date | None = None) -> dict | None:
    record_date = target_date or _extract_record_date(message) or date.today()

    records: list[dict] = []
    weight_match = _WEIGHT_RE.search(message)
    bodyfat_match = _BODYFAT_RE.search(message)

    if weight_match:
        records.append(
            {
                "record_date": record_date.isoformat(),
                "metric_type": "weight",
                "value": float(weight_match.group(1)),
                "unit": "kg",
            }
        )
    if bodyfat_match:
        records.append(
            {
                "record_date": record_date.isoformat(),
                "metric_type": "bodyfat",
                "value": float(bodyfat_match.group(1)),
                "unit": "%",
            }
        )

    if not records:
        fallback = _WEIGHT_FALLBACK_RE.search(message)
        if fallback:
            records.append(
                {
                    "record_date": record_date.isoformat(),
                    "metric_type": "weight",
                    "value": float(fallback.group(1)),
                    "unit": "kg",
                }
            )

    return {"records": records} if records else None


def parse_goal_weight(message: str) -> float | None:
    match = _GOAL_WEIGHT_RE.search(message)
    if not match:
        return None
    return float(match.group(1))


def parse_nutrition_entry(message: str, target_date: date | None = None) -> dict | None:
    record_date = target_date or _extract_record_date(message) or date.today()
    meal_type = _detect_meal_type(message)
    items: list[dict] = []

    for match in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9]+)\s*(\d+(?:\.\d+)?)\s*(g|克|个|ml|毫升)",
        message,
    ):
        name, amount, unit = match.group(1), float(match.group(2)), match.group(3)
        if unit in {"g", "克"}:
            unit = "g"
        ntr = _estimate_nutrients(name, amount, unit)
        items.append(
            {
                "record_date": record_date.isoformat(),
                "meal_type": meal_type,
                "food_name": name,
                "amount": amount,
                "unit": unit,
                "nutrients_snapshot": ntr,
            }
        )

    return {"items": items} if items else None


def _detect_meal_type(message: str) -> str:
    for meal, keywords in MEAL_KEYWORDS.items():
        if any(k in message for k in keywords):
            return meal
    return "lunch"


def _estimate_nutrients(name: str, amount: float, unit: str) -> dict:
    """简单估算 — 一期不做食物库查询。"""
    per_100g = {
        "鸡胸肉": {"cal": 165, "protein": 31, "fat": 3.6, "carb": 0},
        "苹果": {"cal": 52, "protein": 0.3, "fat": 0.2, "carb": 14},
        "鸡蛋": {"cal": 144, "protein": 13, "fat": 10, "carb": 1},
        "米饭": {"cal": 116, "protein": 2.6, "fat": 0.3, "carb": 25.9},
    }
    base = per_100g.get(name, {"cal": 100, "protein": 5, "fat": 3, "carb": 10})
    factor = amount / 100.0 if unit == "g" else 1.0
    if unit == "个" and name == "苹果":
        factor = 1.8
    elif unit == "个" and name == "鸡蛋":
        factor = 0.5
    return {k: round(v * factor, 1) for k, v in base.items()}


def format_body_confirmation(payload: dict) -> str:
    lines = ["请确认以下身体数据写入（source=manual）："]
    for r in payload.get("records", []):
        lines.append(f"- {r['record_date']} {r['metric_type']}: {r['value']} {r['unit']}")
    lines.append("\n回复「确认」写入，或「取消」放弃。")
    return "\n".join(lines)


def format_nutrition_confirmation(payload: dict) -> str:
    lines = ["请确认以下饮食记录写入（source=manual）："]
    for item in payload.get("items", []):
        ntr = item["nutrients_snapshot"]
        lines.append(
            f"- {item['record_date']} {item['meal_type']} {item['food_name']} "
            f"{item['amount']}{item['unit']}：约 {ntr.get('cal', 0)} kcal，"
            f"蛋白 {ntr.get('protein', 0)}g"
        )
    lines.append("\n回复「确认」写入，或「取消」放弃。")
    return "\n".join(lines)
