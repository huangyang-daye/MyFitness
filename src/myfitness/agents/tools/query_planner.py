"""从用户问题解析 DB 查询计划（日期范围 + 数据域）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from myfitness.schemas.state import Intent

from myfitness.graph.context_reflection import needs_personalized_context

_RECENT_DAYS_RE = re.compile(r"最?\s*近\s*(\d+)\s*天")
# 「最近N天 / 过去N天 / 前N天」——要求含数字，避免误吞「前天」
_PAST_DAYS_RE = re.compile(r"(?:最?\s*近|过[去了]?|前)\s*(\d+)\s*天")
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_CN_MD_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_DOT_MD_RE = re.compile(r"(?<!\d)([1-9]\d?)[./](\d{1,2})(?!\d)")
_RANGE_CONNECTOR_RE = re.compile(r"(?:到|至|~|～|-|—|–)")
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
    include_latest_body: bool = False
    muscle_group: str | None = None

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
    if intent == Intent.GENERAL and needs_personalized_context(message):
        return True
    return False


_PROGRESS_END_RE = re.compile(r"(到今天|至今|到目前为止|到目前为止)")
_PROGRESS_HINT_RE = re.compile(r"(进度|趋势|变化|对比|减肥|减脂|增肌|成效|效果)")
# 「今天练背 + 参考过往记录」：今天是要安排的目标日，训练历史应查更宽窗口
_HISTORY_FOR_TRAINING_PLAN_RE = re.compile(
    r"(过往|历史|上次|以往|以前|结合|根据).*(训练|练|记录)|"
    r"训练记录|练.*记录|记录.*练"
)


def build_query_plan(
    message: str,
    intent: Intent,
    domain: str | None = None,
    today: date | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> QueryPlan | None:
    if not needs_database_query(intent, message):
        return None

    today = today or date.today()
    parsed_start, parsed_end = _parse_date_range(message, today, intent)
    plan_start = start_date or parsed_start
    plan_end = end_date or parsed_end
    if plan_start > plan_end:
        plan_start, plan_end = plan_end, plan_start

    domains = _infer_domains(message, intent, domain)
    metric_type = _infer_metric_type(message)
    meal_type = _infer_meal_type(message)
    include_latest_body = needs_personalized_context(message) and intent in {
        Intent.GENERAL,
        Intent.WEB_SEARCH,
        Intent.DATA_QUERY,
        Intent.TREND_ANALYSIS,
    }
    if include_latest_body:
        domains = _ensure_body_domain(domains)

    return QueryPlan(
        start_date=plan_start,
        end_date=plan_end,
        domains=domains,
        metric_type=metric_type,
        meal_type=meal_type,
        include_latest_body=include_latest_body,
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
        month, day = int(m.group(1)), int(m.group(2))
        if _is_valid_month_day(month, day):
            return _resolve_month_day(month, day, today)

    if m := _DOT_MD_RE.search(message):
        month, day = int(m.group(1)), int(m.group(2))
        if _is_valid_month_day(month, day):
            return _resolve_month_day(month, day, today)

    return default


def _is_valid_month_day(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= 31


def parse_date_range_text(
    message: str,
    today: date | None = None,
) -> tuple[date | None, date | None]:
    """从消息解析**连续日期区间**（周期报表 / 趋势图用）。

    优先级：
    1. 显式日期区间（「8月20日到8月25日」「2026-08-20~2026-08-25」）；
    2. 多个离散日期（「8月20号和25号」）→ 取最早到最晚；
    3. 最近/过去/前 N 天 → 含今天的 N 天；
    4. 单日（今天/昨天/8月24日）→ start == end；
    5. 都没有 → (None, None)。
    """
    today = today or date.today()
    tokens = list(_iter_date_tokens(message, today))

    if len(tokens) >= 2:
        start, end = tokens[0][2], tokens[-1][2]
        if start > end:
            start, end = end, start
        return start, end

    if len(tokens) == 1:
        single = tokens[0][2]
        return single, single

    if m := _PAST_DAYS_RE.search(message):
        days = max(int(m.group(1)), 1)
        return today - timedelta(days=days - 1), today

    return None, None


def _iter_date_tokens(message: str, today: date) -> list[tuple[int, int, date]]:
    """按出现顺序提取消息中的日期（ISO / N月N日 / M.D），返回 (起始位置, 结束位置, 日期)。"""
    found: list[tuple[int, int, date]] = []

    def add(start: int, end: int, value: date) -> None:
        if not any(f[2] == value for f in found):
            found.append((start, end, value))

    for m in _ISO_DATE_RE.finditer(message):
        try:
            add(m.start(), m.end(), date.fromisoformat(m.group(1)))
        except ValueError:
            continue

    for m in _CN_MD_RE.finditer(message):
        month, day = int(m.group(1)), int(m.group(2))
        if not _is_valid_month_day(month, day):
            continue
        try:
            add(m.start(), m.end(), _resolve_month_day(month, day, today))
        except ValueError:
            continue

    for m in _DOT_MD_RE.finditer(message):
        month, day = int(m.group(1)), int(m.group(2))
        if not _is_valid_month_day(month, day):
            continue
        try:
            add(m.start(), m.end(), _resolve_month_day(month, day, today))
        except ValueError:
            continue

    found.sort(key=lambda t: t[0])
    return found


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

    # 「…到今天 / 至今 + 进度/趋势」表示区间终点是今天，不是只查今天
    if intent == Intent.TREND_ANALYSIS and (
        _PROGRESS_END_RE.search(message) or _PROGRESS_HINT_RE.search(message)
    ):
        if "今天" in message or "今日" in message:
            explicit = parse_date_range_text(message, today)
            if explicit[0] is not None:
                return explicit[0], explicit[1] or today
            return today - timedelta(days=29), today

    single = parse_single_date(message, today)
    if single is not None and intent != Intent.TREND_ANALYSIS:
        if _needs_training_history_window(message):
            return today - timedelta(days=29), today
        return single, single

    if single is not None and intent == Intent.TREND_ANALYSIS:
        # 趋势类里单独的「今天/8月24日」仍可能是单日快照查询
        if not (_PROGRESS_END_RE.search(message) or _PROGRESS_HINT_RE.search(message)):
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
    if "体重" in message or re.search(r"\d+\s*(?:kg|公斤)", message, re.IGNORECASE):
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


def _ensure_body_domain(domains: tuple[str, ...]) -> tuple[str, ...]:
    if "body" in domains:
        return domains
    return ("body", *domains)


def _needs_training_history_window(message: str) -> bool:
    """今天安排训练计划，但需参考历史记录时，不应把查询范围缩成单日。"""
    if not _HISTORY_FOR_TRAINING_PLAN_RE.search(message):
        return False
    return any(k in message for k in TRAINING_KEYWORDS) or "练" in message
