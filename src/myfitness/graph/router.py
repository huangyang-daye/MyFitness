"""Router — 意图识别（关键词兜底 + 可选 LLM，受熔断守卫约束）。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from myfitness.llm.factory import is_llm_configured
from myfitness.schemas.state import Intent, PendingConfirmation

logger = logging.getLogger(__name__)

CONFIRM_WORDS = {"确认", "确定", "是的", "好", "ok", "yes", "写入", "保存"}
CANCEL_WORDS = {"取消", "不要", "算了", "no", "cancel"}


@dataclass
class RouteResult:
    intent: Intent
    domain: str | None = None  # body | nutrition | fitness
    confirmation_action: str | None = None  # confirm | cancel


def classify_intent(
    message: str,
    pending: PendingConfirmation | None = None,
) -> RouteResult:
    text = message.strip()
    lower = text.lower()

    if pending and _is_confirmation_response(text):
        if any(w in lower or w in text for w in CONFIRM_WORDS):
            return RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="confirm")
        if any(w in lower or w in text for w in CANCEL_WORDS):
            return RouteResult(Intent.CONFIRMATION_RESPONSE, confirmation_action="cancel")

    keyword = _keyword_classify(text)
    if keyword:
        return keyword

    if is_llm_configured():
        llm_result = _llm_classify(text)
        if llm_result:
            return llm_result

    return RouteResult(Intent.GENERAL)


def _is_confirmation_response(text: str) -> bool:
    lower = text.lower()
    return (
        any(w in lower or w in text for w in CONFIRM_WORDS)
        or any(w in lower or w in text for w in CANCEL_WORDS)
        or text in {"y", "n"}
    )


def _keyword_classify(text: str) -> RouteResult | None:
    if re.search(r"(同步|拉取|更新).*(训记|数据)", text):
        return RouteResult(Intent.SYNC_TRIGGER)

    if re.search(r"(记录|录入|添加).*(体重|体脂)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="body")
    if re.search(r"(记录|录入|添加).*(食物|餐|吃|早餐|午餐|晚餐|零食)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")
    if re.search(r"(吃了|午餐|晚餐|早餐|零食)", text) and re.search(r"\d+\s*(g|克|个)", text):
        return RouteResult(Intent.MANUAL_ENTRY, domain="nutrition")

    if re.search(r"(改成|调整|取消).*(训练|计划|休息)", text):
        return RouteResult(Intent.PLAN_ADJUST, domain="fitness")

    if re.search(r"(近\s*\d+\s*天|趋势|变化|对比)", text):
        return RouteResult(Intent.TREND_ANALYSIS)

    if re.search(r"(目标|降到|增到|减到).*(kg|公斤|%)", text, re.I):
        return RouteResult(Intent.GOAL_SETTING, domain="body")

    if re.search(r"(多少|查询|昨天|今天).*(蛋白|热量|体重|训练|吃)", text):
        return RouteResult(Intent.DATA_QUERY)

    if re.search(r"(多少|查询|昨天|今天)", text):
        return RouteResult(Intent.DATA_QUERY)

    return None


def _llm_classify(text: str) -> RouteResult | None:
    """LLM 意图分类：受熔断守卫约束，失败静默回退 None（走 general）。"""
    from myfitness.llm.guard import LlmCircuitOpenError, get_llm_guard

    guard = get_llm_guard()
    try:
        guard.acquire()
    except LlmCircuitOpenError:
        logger.info("LLM 熔断中，意图分类走关键词兜底")
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from myfitness.llm.factory import get_llm

        llm = get_llm()
        prompt = (
            "你是意图分类器。根据用户消息返回 JSON："
            '{"intent":"data_query|manual_entry|plan_adjust|trend_analysis|goal_setting|sync_trigger|general",'
            '"domain":"body|nutrition|fitness|null"}'
        )
        resp = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=text),
            ]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            guard.record_failure("empty classify response")
            return None
        data = json.loads(match.group())
        intent = Intent(data.get("intent", "general"))
        domain = data.get("domain")
        if domain == "null":
            domain = None
        guard.record_success()
        return RouteResult(intent, domain=domain)
    except Exception as exc:
        guard.record_failure(str(exc))
        return None


def agents_for_intent(intent: Intent, domain: str | None = None) -> list[str]:
    mapping: dict[Intent, list[str]] = {
        Intent.DATA_QUERY: ["body", "nutrition", "fitness"],
        Intent.MANUAL_ENTRY: [domain or "nutrition"],
        Intent.PLAN_ADJUST: [domain or "fitness"],
        Intent.TREND_ANALYSIS: ["body", "nutrition", "fitness"],
        Intent.GOAL_SETTING: ["body"],
        Intent.SYNC_TRIGGER: [],
        Intent.GENERAL: [],
        Intent.CONFIRMATION_RESPONSE: [],
    }
    return mapping.get(intent, [])
