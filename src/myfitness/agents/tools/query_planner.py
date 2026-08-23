"""从用户问题解析 DB 查询计划（日期范围 + 数据域）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from myfitness.schemas.state import Intent

_RECENT_DAYS_RE = re.compile(r"最?\s*近\s*(\d+)\s*天")
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_CN_MD_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_DOT_MD_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?!\d)")
BODY_KEYWORDS = ("体重", "体脂", "围度", "公斤", "kg")
NUTRITION_KEYWORDS = ("蛋白", "热量", "饮食", "吃了", "吃", "餐", "kcal", "卡路里", "碳水", "脂肪", "营养")
TRAINING_KEYWORDS = ("训练", "练", "卧推", "深蹲", "硬拉", "动作", "组", "健身")


@dataclass(frozen=True)
class QueryPlan:
    start_date: date
    end_date: date
    domains: tuple[str, ...]
    metric_type: str | None = None
    meal_type: str | None = None

    @property
    def lookback_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


def needs_database_query(intent: Intent, message: str) -> bool:
    if intent in {
        Intent.DATA_QUERY,
        Intent.TREND_ANALYSIS,
        Intent.GOAL_SETTING,
        Intent.PLAN_ADJUST,
    }:
        return True
    if intent == Intent.GENERAL and _has_data_keywords(message):
        return True
    return False


def build_query_plan(
    message: str,
    intent: Intent,
    domain: str | None = None,
    today: date | None = None,
) -> QueryPlan | None:
    if not needs_database_query(intent, message):
        return None

    today = today or date.today()
    start, end = _parse_date_range(message, today, intent)
    domains = _infer_domains(message, intent, domain)
    metric_type = _infer_metric_type(message)
    meal_type = _infer_meal_type(message)

    return QueryPlan(
        start_date=start,
        end_date=end,
        domains=domains,
        metric_type=metric_type,
        meal_type=meal_type,
    )


def _has_data_keywords(message: str) -> bool:
    keywords = BODY_KEYWORDS + NUTRITION_KEYWORDS + TRAINING_KEYWORDS
    return any(k in message for k in keywords) or bool(
        _RECENT_DAYS_RE.search(message)
        or re.search(r"(多少|查询|昨天|今天|前天|\d{4}-\d{2}-\d{2})", message)
    )


def parse_single_date(
    message: str,
    today: date | None = None,
    *,
    default: date | None = None,
) -> date | None:
    """从用户消息解析单个日期（日报、录入等）。"""
    today = today or date.today()

    if "今天" in message:
        return today
    if "昨天" in message:
        return today - timedelta(days=1)
    if "前天" in message:
        return today - timedelta(days=2)

    if m := _ISO_DATE_RE.search(message):
        return date.fromisoformat(m.group(1))

    if m := _CN_MD_RE.search(message):
        return _resolve_month_day(int(m.group(1)), int(m.group(2)), today)

    if m := _DOT_MD_RE.search(message):
        return _resolve_month_day(int(m.group(1)), int(m.group(2)), today)

    return default


def _resolve_month_day(month: int, day: int, today: date) -> date:
    year = today.year
    try:
        candidate = date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"无效日期：{month}月{day}日") from exc
    if candidate > today:
        candidate = date(year - 1, month, day)
    return candidate


def _parse_date_range(message: str, today: date, intent: Intent) -> tuple[date, date]:
    if m := _RECENT_DAYS_RE.search(message):
        days = int(m.group(1))
        # 含今天：近 7 天 = today-6 … today
        return today - timedelta(days=days - 1), today

    single = parse_single_date(message, today)
    if single is not None:
        return single, single

    if intent == Intent.TREND_ANALYSIS:
        return today - timedelta(days=29), today

    # 默认查最近 7 天（含今天）
    return today - timedelta(days=6), today


def _infer_domains(message: str, intent: Intent, domain: str | None) -> tuple[str, ...]:
    if domain:
        return (domain,)

    domains: list[str] = []
    if any(k in message for k in BODY_KEYWORDS):
        domains.append("body")
    if any(k in message for k in NUTRITION_KEYWORDS):
        domains.append("nutrition")
    if any(k in message for k in TRAINING_KEYWORDS):
        domains.append("training")

    if domains:
        return tuple(domains)

    if intent in {Intent.TREND_ANALYSIS, Intent.DATA_QUERY}:
        return ("body", "nutrition", "training")
    if intent == Intent.GOAL_SETTING:
        return ("body",)
    return ("body", "nutrition", "training")


def _infer_metric_type(message: str) -> str | None:
    if "体脂" in message:
        return "bodyfat"
    if "体重" in message or re.search(r"\d+\s*(?:kg|公斤)", message, re.I):
        return "weight"
    return None


def _infer_meal_type(message: str) -> str | None:
    mapping = {
        "早餐": "breakfast",
        "午饭": "lunch",
        "午餐": "lunch",
        "晚饭": "dinner",
        "晚餐": "dinner",
        "零食": "snack",
    }
    for kw, meal in mapping.items():
        if kw in message:
            return meal
    return None
