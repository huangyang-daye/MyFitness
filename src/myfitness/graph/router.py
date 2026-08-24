"""Router — 意图识别：LLM 意图 Agent 优先，关键词规则兜底。

分类顺序：
1. 待确认上下文的确认/取消匹配（无需 LLM）；
2. LLM 意图识别 Agent（agents/intent_agent.py，受熔断守卫约束）；
3. 关键词/正则规则兜底（支持多意图与日期解析）；
4. 默认 GENERAL。
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from myfitness.agents.tools.query_planner import parse_single_date
from myfitness.schemas.state import Intent, PendingConfirmation, RouteResult

logger = logging.getLogger(__name__)

CONFIRM_WORDS = {"确认", "确定", "是的", "好", "ok", "yes", "写入", "保存"}
CANCEL_WORDS = {"取消", "不要", "算了", "no", "cancel"}

_SYNC_RE = re.compile(r"(同步|拉取|更新).*(训记|数据)")
_RECENT_DAYS_RE = re.compile(r"最?\s*近\s*(\d+)\s*天")
_SCHEDULE_WORDS = ("定时", "每天", "每日", "自动")
_SCHEDULE_ACTION_WORDS = ("日报", "同步", "任务", "报告")


def classify_intent(
    message: str,
    pending: PendingConfirmation | None = None,
    *,
    today: date | None = None,
    use_llm: bool = True,
) -> RouteResult:
    """识别用户消息的意图（支持多意图）与日期范围。

    - use_llm=True 时优先调用 LLM 意图 Agent，失败或未配置回退关键词规则；
    - use_llm=False 直接走关键词规则（测试/评估关键词层时使用）。
    """
    text = message.strip()
    lower = text.lower()
    today = today or date.today()

    # ① 有待确认操作时，先匹配确认/取消词（无歧义，无需 LLM）
    if pending and _is_confirmation_response(text):
        if any(w in lower or w in text for w in CONFIRM_WORDS):
            return RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="confirm")
        if any(w in lower or w in text for w in CANCEL_WORDS):
            return RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="cancel")

    keyword = _keyword_classify(text, today)

    # ② LLM 意图识别 Agent 优先
    if use_llm:
        llm_result = _llm_classify(text, today)
        if llm_result is not None:
            return _reconcile(llm_result, keyword)
        logger.info("意图 Agent 未返回有效结果，使用关键词匹配兜底")

    # ③ 关键词兜底
    if keyword:
        return keyword

    return RouteResult(intents=[Intent.GENERAL])


def _reconcile(llm: RouteResult, keyword: RouteResult | None) -> RouteResult:
    """LLM 结果与关键词结果的调和。

    - LLM 判为 general 而关键词有明确命中 → 采用关键词（防 LLM 过度泛化）；
    - LLM 未提取到日期而关键词解析出了日期（同步/日报场景）→ 补齐日期；
    - 其余情况信任 LLM。
    """
    if keyword is not None:
        if llm.intent == Intent.GENERAL and keyword.intent != Intent.GENERAL:
            return keyword
        if llm.start_date is None and keyword.start_date is not None and (
            llm.has(Intent.SYNC_TRIGGER) or llm.has(Intent.REPORT_TRIGGER)
        ):
            llm.start_date = keyword.start_date
            llm.end_date = keyword.end_date
    return llm


def _is_confirmation_response(text: str) -> bool:
    lower = text.lower()
    return (
        any(w in lower or w in text for w in CONFIRM_WORDS)
        or any(w in lower or w in text for w in CANCEL_WORDS)
        or text in {"y", "n"}
    )


def _is_report_request(text: str) -> bool:
    """一次性日报请求（排除定时/每天/每日/自动等重复任务表达）。"""
    if any(k in text for k in _SCHEDULE_WORDS):
        return False
    return any(k in text for k in ("日报", "晨报")) or bool(
        re.search(r"生成.*(?:日报|报告|晨报)", text)
    )


def _keyword_classify(text: str, today: date | None = None) -> RouteResult | None:
    today = today or date.today()

    # 定时任务（重复性任务管理优先级最高，防止「每天…生成日报」误判为一次性日报）
    if any(k in text for k in ("定时任务", "查看定时", "取消定时", "停用定时")) or (
        any(k in text for k in ("定时", "每天", "每日"))
        and any(k in text for k in _SCHEDULE_ACTION_WORDS)
    ):
        return RouteResult(intents=[Intent.SCHEDULE_MANAGE])

    # 可组合的动作意图：同步 + 日报（「同步8月24日数据并生成日报」→ 先同步再出报告）
    action_intents: list[Intent] = []
    if _SYNC_RE.search(text):
        action_intents.append(Intent.SYNC_TRIGGER)
    if _is_report_request(text):
        action_intents.append(Intent.REPORT_TRIGGER)

    if action_intents:
        start, end = _parse_action_date_range(text, today)
        return RouteResult(
            intents=action_intents,
            domain=_infer_domain_from_text(text),
            start_date=start,
            end_date=end,
        )

    # 单一意图规则
    if re.search(r"(记录|录入|添加).*(体重|体脂)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="body")
    if re.search(r"(记录|录入|添加).*(食物|餐|吃|早餐|午餐|晚餐|零食)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")
    if re.search(r"(吃了|午餐|晚餐|早餐|零食)", text) and re.search(r"\d+\s*(g|克|个)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")

    if re.search(r"(改成|调整|取消).*(训练|计划|休息)", text):
        return RouteResult(Intent.PLAN_ADJUST, domain="fitness")

    if re.search(r"(最?\s*近\s*\d+\s*天|近\s*\d+\s*天|趋势|变化|对比)", text):
        return RouteResult(Intent.TREND_ANALYSIS)

    if re.search(r"(最?\s*近\s*\d+\s*天|近\s*\d+\s*天).*(蛋白|热量|体重|训练|吃)", text):
        return RouteResult(Intent.DATA_QUERY, domain=_infer_domain_from_text(text))

    if re.search(r"(目标|降到|增到|减到).*(kg|公斤|%)", text, re.I):
        return RouteResult(Intent.GOAL_SETTING, domain="body")

    if re.search(r"(多少|查询|昨天|今天).*(蛋白|热量|体重|训练|吃)", text):
        return RouteResult(Intent.DATA_QUERY, domain=_infer_domain_from_text(text))

    if re.search(r"(多少|查询|昨天|今天)", text):
        return RouteResult(Intent.DATA_QUERY)

    return None


def _parse_action_date_range(text: str, today: date) -> tuple[date | None, date | None]:
    """从同步/日报类消息解析日期范围；未指明日期时返回 (None, None)。"""
    if "今天" in text or "今日" in text:
        return today, today
    if "昨天" in text or "昨日" in text:
        return today - timedelta(days=1), today - timedelta(days=1)
    if "前天" in text:
        return today - timedelta(days=2), today - timedelta(days=2)

    if m := _RECENT_DAYS_RE.search(text):
        days = int(m.group(1))
        if days > 0:
            # 「最近N天」含今天
            return today - timedelta(days=days - 1), today

    # N月N日 / YYYY-MM-DD / M.D 等单日表达
    single = parse_single_date(text, today)
    if single is not None:
        return single, single

    return None, None


def _infer_domain_from_text(text: str) -> str | None:
    if any(k in text for k in ("体重", "体脂", "围度", "公斤", "kg")):
        return "body"
    if any(k in text for k in ("蛋白", "热量", "饮食", "吃了", "餐", "卡路里", "碳水")):
        return "nutrition"
    if any(k in text for k in ("训练", "练", "卧推", "深蹲", "硬拉", "健身")):
        return "fitness"
    return None


def _llm_classify(text: str, today: date) -> RouteResult | None:
    """LLM 意图识别 Agent：未配置/熔断/解析失败时返回 None（走关键词兜底）。"""
    from myfitness.agents.intent_agent import run_intent_agent

    return run_intent_agent(text, today=today)


def agents_for_intent(intent: Intent, domain: str | None = None) -> list[str]:
    mapping: dict[Intent, list[str]] = {
        Intent.DATA_QUERY: ["body", "nutrition", "fitness"],
        Intent.MANUAL_ENTRY: [domain or "nutrition"],
        Intent.PLAN_ADJUST: [domain or "fitness"],
        Intent.TREND_ANALYSIS: ["body", "nutrition", "fitness"],
        Intent.GOAL_SETTING: ["body"],
        Intent.SYNC_TRIGGER: [],
        Intent.SCHEDULE_MANAGE: [],
        Intent.REPORT_TRIGGER: [],
        Intent.GENERAL: [],
        Intent.CONFIRMATION_RESPONSE: [],
    }
    return mapping.get(intent, [])
