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
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from myfitness.agents.tools.chart_tools import INSERT_DOC_KEYWORDS, is_chart_request
from myfitness.agents.tools.document_tools import is_document_generation_request
from myfitness.agents.tools.web_search import is_web_search_request
from myfitness.debug import log_intent_result
from myfitness.schemas.state import Intent, PendingConfirmation, RouteResult

logger = logging.getLogger(__name__)

CONFIRM_WORDS = {"确认", "确定", "是的", "好", "ok", "yes", "写入", "保存"}
CANCEL_WORDS = {"取消", "不要", "算了", "no", "cancel"}

_SYNC_RE = re.compile(r"(同步|拉取|更新).*(训记|数据)")
_RECENT_DAYS_RE = re.compile(r"最?\s*近\s*(\d+)\s*天")
_PAST_DAYS_RE = re.compile(r"(?:过[去了]?|前)\s*(\d+)\s*天")
_SCHEDULE_WORDS = ("定时", "每天", "每日", "自动")
_SCHEDULE_ACTION_WORDS = ("日报", "同步", "任务", "报告")
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


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
    today = today or datetime.now(_LOCAL_TZ).date()

    # ① 有待确认操作时，先匹配确认/取消词（无歧义，无需 LLM）
    if pending and _is_confirmation_response(text):
        if any(w in lower or w in text for w in CONFIRM_WORDS):
            result = RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="confirm")
            log_intent_result(text, result, source="confirmation_rule")
            return result
        if any(w in lower or w in text for w in CANCEL_WORDS):
            result = RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="cancel")
            log_intent_result(text, result, source="confirmation_rule")
            return result

    keyword = _keyword_classify(text, today)

    # ② LLM 意图识别 Agent 优先
    if use_llm:
        llm_result = _llm_classify(text, today)
        if llm_result is not None:
            result = _reconcile(llm_result, keyword)
            log_intent_result(text, result, source="llm_reconciled")
            return result
        logger.info("意图 Agent 未返回有效结果，使用关键词匹配兜底")

    # ③ 关键词兜底
    if keyword:
        log_intent_result(text, keyword, source="keyword")
        return keyword

    result = RouteResult(intents=[Intent.GENERAL])
    log_intent_result(text, result, source="default")
    return result


def _reconcile(llm: RouteResult, keyword: RouteResult | None) -> RouteResult:
    """LLM 结果与关键词结果的调和。

    - LLM 判为 general 而关键词有明确命中 → 采用关键词（防 LLM 过度泛化）；
    - LLM 未提取到日期而关键词解析出了日期（同步/日报场景）→ 补齐日期；
    - LLM 解析出的日期范围比关键词**更窄**、关键词范围为其超集（如「昨天和今天」
      LLM 只给今天）时，拓宽到关键词范围，确保不漏掉用户提到的日期；
    - 其余情况信任 LLM。
    """
    if keyword is not None:
        if llm.intent == Intent.GENERAL and keyword.intent != Intent.GENERAL:
            return keyword
        if (
            keyword.has(Intent.WEB_SEARCH)
            and llm.intent == Intent.DATA_QUERY
            and llm.start_date is None
        ):
            return keyword
        if (
            llm.has(Intent.REPORT_TRIGGER)
            and keyword.has(Intent.TREND_ANALYSIS)
            and keyword.domain
        ):
            return keyword
        handles_date_range = (
            llm.has(Intent.SYNC_TRIGGER)
            or llm.has(Intent.REPORT_TRIGGER)
            or llm.has(Intent.CHART_TRIGGER)
        )
        llm_missed_date = llm.start_date is None and keyword.start_date is not None
        keyword_is_wider = bool(
            llm.start_date
            and llm.end_date
            and keyword.start_date
            and keyword.end_date
            and keyword.start_date <= llm.start_date
            and keyword.end_date >= llm.end_date
            and (keyword.start_date < llm.start_date or keyword.end_date > llm.end_date)
        )
        if handles_date_range and (llm_missed_date or keyword_is_wider):
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
    """一次性完整日报请求（排除定时/每天/每日/自动、领域专项报告、纯插入图表）。"""
    if is_document_generation_request(text):
        return False
    if any(k in text for k in _SCHEDULE_WORDS):
        return False
    if _is_focused_topic_report(text):
        return False
    if (
        is_chart_request(text)
        and any(k in text for k in INSERT_DOC_KEYWORDS)
        and "生成" not in text
    ):
        return False
    return any(k in text for k in ("日报", "晨报", "报表")) or bool(
        re.search(r"生成.*(?:日报|报告|晨报|报表)", text)
    )


def _is_focused_topic_report(text: str) -> bool:
    """领域/主题明确的专项报告或趋势分析，走 trend_analysis 而非完整日报。"""
    if any(k in text for k in _SCHEDULE_WORDS):
        return False
    if any(k in text for k in ("日报", "晨报", "综合报告", "完整报告", "健康报告")):
        return False
    has_topic_focus = bool(
        re.search(
            r"(变化|趋势|对比|分析).*(报告|分析)|(?:报告|分析).*(变化|趋势|对比|分析)",
            text,
        )
    )
    # 「生成 + 日期/区间 + 报告」= 完整周期报告（即使后面提到某指标折线图）
    if re.search(r"生成\s*.+(?:报告|日报)", text) and not has_topic_focus:
        return False

    domain = _infer_domain_from_text(text)
    if not domain:
        return False
    if has_topic_focus:
        return True
    # 「近N天 + 领域 + 报告/分析」如「近7天体重报告」
    if re.search(r"近\s*\d+\s*天", text) and re.search(r"(报告|分析)", text):
        return True
    return False


def _keyword_classify(text: str, today: date | None = None) -> RouteResult | None:
    today = today or datetime.now(_LOCAL_TZ).date()

    # 定时任务（重复性任务管理优先级最高，防止「每天…生成日报」误判为一次性日报）
    if any(k in text for k in ("定时任务", "查看定时", "取消定时", "停用定时")) or (
        any(k in text for k in ("定时", "每天", "每日"))
        and any(k in text for k in _SCHEDULE_ACTION_WORDS)
    ):
        return RouteResult(intents=[Intent.SCHEDULE_MANAGE])

    # 「把体重折线图插入到8月24日的日报」只是插入图表，不该触发生成新日报；
    # 但「生成日报并插入趋势图」需要同时触发 report + chart。
    insert_only = (
        is_chart_request(text)
        and any(k in text for k in INSERT_DOC_KEYWORDS)
        and not _is_report_request(text)
    )

    # 主题文档生成（饮食规划、训练计划等）——不是日报，走分析 + 写文档
    if is_document_generation_request(text) and not is_chart_request(text):
        domain = _infer_domain_from_text(text)
        if domain is None and re.search(r"饮食|营养|餐|规划", text):
            domain = "nutrition"
        if domain is None and re.search(r"训练|健身", text):
            domain = "fitness"
        return RouteResult(
            intents=[Intent.TREND_ANALYSIS if domain else Intent.GENERAL],
            domain=domain,
        )

    # 领域专项报告 / 趋势分析（优先于完整日报）
    if _is_focused_topic_report(text):
        start, end = _parse_action_date_range(text, today)
        return RouteResult(
            intents=[Intent.TREND_ANALYSIS],
            domain=_infer_domain_from_text(text),
            start_date=start,
            end_date=end,
        )

    # 可组合的动作意图：同步 + 日报 + 统计图（按顺序执行）
    action_intents: list[Intent] = []
    if _SYNC_RE.search(text):
        action_intents.append(Intent.SYNC_TRIGGER)
    if _is_report_request(text) and not insert_only:
        action_intents.append(Intent.REPORT_TRIGGER)
    if is_chart_request(text):
        action_intents.append(Intent.CHART_TRIGGER)

    if action_intents:
        start, end = _parse_action_date_range(text, today)
        return RouteResult(
            intents=action_intents,
            domain=_infer_domain_from_text(text),
            start_date=start,
            end_date=end,
        )

    # 单一意图规则
    if re.search(r"(记录|录入|添加|初始).*(体重|体脂)", text) or (
        re.search(r"(记录|录入|初始)", text) and re.search(r"(体重|体脂)", text)
    ):
        intents: list[Intent] = [Intent.MANUAL_ENTRY]
        start, end = _parse_action_date_range(text, today)
        if re.search(r"(目标|减到|降到|增到).*(kg|公斤|千克|%)", text, re.I):
            intents.append(Intent.GOAL_SETTING)
        if re.search(r"(评价|进度|怎么样|分析|趋势|变化)", text):
            intents.append(Intent.TREND_ANALYSIS)
        return RouteResult(
            intents=intents,
            domain="body",
            start_date=start,
            end_date=end or today,
        )
    if re.search(r"(记录|录入|添加).*(食物|餐|吃|早餐|午餐|晚餐|零食)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")
    if re.search(r"(吃了|午餐|晚餐|早餐|零食)", text) and re.search(r"\d+\s*(g|克|个)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")

    if re.search(r"(改成|调整|取消).*(训练|计划|休息)", text):
        return RouteResult(Intent.PLAN_ADJUST, domain="fitness")

    # 统计图（折线图 / 柱状图 / 趋势图…）：交由 chart tool 渲染 mermaid，
    # 与 trend_analysis 区分（后者是文字分析，前者要出图）
    if is_chart_request(text):
        start, end = _parse_action_date_range(text, today)
        return RouteResult(
            intents=[Intent.CHART_TRIGGER],
            domain=_infer_domain_from_text(text),
            start_date=start,
            end_date=end,
        )

    # 联网检索须在 data_query / trend 兜底之前，避免「搜一下今天HIIT怎么练」被「今天」吞掉
    if is_web_search_request(text):
        return RouteResult(Intent.WEB_SEARCH, domain=_infer_domain_from_text(text))

    if re.search(r"(最?\s*近\s*\d+\s*天|近\s*\d+\s*天|趋势|变化|对比)", text):
        start, end = _parse_action_date_range(text, today)
        return RouteResult(
            Intent.TREND_ANALYSIS,
            domain=_infer_domain_from_text(text),
            start_date=start,
            end_date=end,
        )

    if re.search(r"(最?\s*近\s*\d+\s*天|近\s*\d+\s*天).*(蛋白|热量|体重|训练|吃)", text):
        return RouteResult(Intent.DATA_QUERY, domain=_infer_domain_from_text(text))

    if re.search(r"(目标|降到|增到|减到).*(kg|公斤|%)", text, re.IGNORECASE):
        return RouteResult(Intent.GOAL_SETTING, domain="body")

    if re.search(r"(多少|查询|昨天|今天).*(蛋白|热量|体重|训练|吃)", text):
        return RouteResult(Intent.DATA_QUERY, domain=_infer_domain_from_text(text))

    if re.search(r"(多少|查询|昨天|今天)", text):
        return RouteResult(Intent.DATA_QUERY)

    return None


def _parse_action_date_range(text: str, today: date) -> tuple[date | None, date | None]:
    """从同步/日报类消息解析日期范围；未指明日期时返回 (None, None)。

    一条消息可能包含多个日期点（如「昨天和今天」「前天到昨天」
    「8月20号和25号」），此时取所有提及日期的**最早与最晚**构成连续范围，
    而不是碰到「今天」就只返回当天。
    """
    # 收集所有提及的日期区间（每个元素为 (start, end)）
    candidates: list[tuple[date, date]] = []

    # 相对日词（每个是一个单日点）
    if "今天" in text or "今日" in text:
        candidates.append((today, today))
    if "昨天" in text or "昨日" in text:
        candidates.append((today - timedelta(days=1), today - timedelta(days=1)))
    if "前天" in text:
        candidates.append((today - timedelta(days=2), today - timedelta(days=2)))

    # 最近N天 / 近N天 / 过去N天 / 前N天 → 含今天的范围
    for pattern in (_RECENT_DAYS_RE, _PAST_DAYS_RE):
        if m := pattern.search(text):
            days = int(m.group(1))
            if days > 0:
                candidates.append((today - timedelta(days=days - 1), today))
            break

    # 显式日期（N月N日 / YYYY-MM-DD / M.D）：消息中可能多个，逐个作为单日点
    for d in _iter_explicit_dates(text, today):
        candidates.append((d, d))

    if not candidates:
        return None, None

    start = min(c[0] for c in candidates)
    end = max(c[1] for c in candidates)
    return start, end


def _iter_explicit_dates(text: str, today: date) -> Iterator[date]:
    """提取消息中所有显式单日日期（N月N日 / YYYY-MM-DD / M.D）。"""
    from myfitness.agents.tools.query_planner import _resolve_month_day

    seen: set[str] = set()
    # YYYY-MM-DD
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        key = d.isoformat()
        if key not in seen:
            seen.add(key)
            yield d
    # N月N日 / N月N号
    for m in re.finditer(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", text):
        month, day = int(m.group(1)), int(m.group(2))
        try:
            d = _resolve_month_day(month, day, today)
        except ValueError:
            continue
        key = d.isoformat()
        if key not in seen:
            seen.add(key)
            yield d
    # M.D / M/D（点号分隔，避免与小数体重等混淆：要求整体像日期）
    for m in re.finditer(r"(?<!\d)([1-9]\d?)[./](\d{1,2})(?!\d)", text):
        month, day = int(m.group(1)), int(m.group(2))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        try:
            d = _resolve_month_day(month, day, today)
        except ValueError:
            continue
        key = d.isoformat()
        if key not in seen:
            seen.add(key)
            yield d


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
        Intent.CHART_TRIGGER: [],
        Intent.WEB_SEARCH: [],
        Intent.GENERAL: [],
        Intent.CONFIRMATION_RESPONSE: [],
    }
    return mapping.get(intent, [])
