"""上下文反思 — 在作答前核查个体数据是否已从数据库检索并确认。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from myfitness.graph.task_plan import ExecutionResult
from myfitness.llm.factory import chat_completion, is_llm_configured

logger = logging.getLogger(__name__)

_PERSONALIZED_RE = re.compile(
    r"(减脂|增肌|减重|增重|塑形|饮食|热量|卡路里|蛋白|营养|食堂|忌口|"
    r"我的|根据我|为我|个性化|安排|计划|目标)",
    re.IGNORECASE,
)
_BODY_FACT_RE = re.compile(
    r"(体重|体脂|公斤|kg|当前|最新|起点|初始|减到|降到|目标)",
    re.IGNORECASE,
)


@dataclass
class ContextReflection:
    """作答前的事实就绪性评估。"""

    ready: bool
    feedback: str = ""
    missing_fetches: list[str] = field(default_factory=list)
    confirmed_notes: str = ""
    retry_task_ids: list[str] = field(default_factory=list)


def needs_personalized_context(message: str) -> bool:
    """问题是否需要结合用户个体数据作答。"""
    text = message.strip()
    if len(text) < 4:
        return False
    return bool(_PERSONALIZED_RE.search(text))


def needs_body_metrics_confirmation(message: str) -> bool:
    """个性化问题是否可能涉及体重/体脂等需数据库确认的身体指标。"""
    if not needs_personalized_context(message):
        return False
    return bool(_BODY_FACT_RE.search(message))


def reflect_before_answer(
    user_message: str,
    execution: ExecutionResult,
    *,
    fetch_task_ids: list[str] | None = None,
    use_llm: bool | None = None,
) -> ContextReflection:
    """作答前反思：个体数据是否已从 DB 检索，能否安全引用。"""
    if not needs_body_metrics_confirmation(user_message):
        return ContextReflection(ready=True)

    llm_enabled = is_llm_configured() if use_llm is None else use_llm
    if llm_enabled:
        llm_result = _llm_reflect(user_message, execution, fetch_task_ids=fetch_task_ids or [])
        if llm_result is not None:
            return llm_result

    return _rule_reflect(user_message, execution, fetch_task_ids=fetch_task_ids or [])


def _has_db_latest_weight(execution: ExecutionResult) -> bool:
    context = execution.context
    if context is None:
        return False
    body = (context.query_results or {}).get("body") or {}
    weight = (body.get("latest_metrics") or {}).get("weight")
    if isinstance(weight, dict) and weight.get("value") is not None:
        return True
    if execution.agent_outputs.body and execution.agent_outputs.body.current_metrics.weight_kg is not None:
        return True
    return False


def _rule_reflect(
    user_message: str,
    execution: ExecutionResult,
    *,
    fetch_task_ids: list[str],
) -> ContextReflection:
    if _has_db_latest_weight(execution):
        notes = _build_confirmed_notes(execution)
        return ContextReflection(ready=True, confirmed_notes=notes)

    feedback = (
        "个性化建议需要先检索并确认数据库中的最新身体指标，"
        "不能仅凭画像或语义检索中的「初始体重」推断当前体重。"
    )
    retry_ids = fetch_task_ids or [
        result.task_id
        for result in execution.task_results
        if result.intent.value == "data_query"
    ]
    return ContextReflection(
        ready=False,
        feedback=feedback,
        missing_fetches=["body_latest"],
        retry_task_ids=retry_ids,
    )


def _build_confirmed_notes(execution: ExecutionResult) -> str:
    context = execution.context
    if context is None:
        return ""
    lines: list[str] = []
    body = (context.query_results or {}).get("body") or {}
    weight = (body.get("latest_metrics") or {}).get("weight")
    if isinstance(weight, dict) and weight.get("value") is not None:
        lines.append(
            f"已确认最新体重 {weight['value']}{weight.get('unit', 'kg')} "
            f"（{weight.get('date')}，来源 {weight.get('source', '数据库')}）"
        )
    for goal in context.user_goals:
        if goal.get("goal_type") != "weight":
            continue
        if goal.get("start_value") is not None:
            lines.append(f"目标起点体重 {goal['start_value']} kg（{goal.get('start_date', '')}）")
        if goal.get("target_value") is not None:
            lines.append(f"目标体重 {goal['target_value']} kg")
    return "；".join(lines)


def _llm_reflect(
    user_message: str,
    execution: ExecutionResult,
    *,
    fetch_task_ids: list[str],
) -> ContextReflection | None:
    context = execution.context
    payload = {
        "user_message": user_message,
        "query_results": context.query_results if context else {},
        "user_goals": context.user_goals if context else [],
        "agent_summaries": _agent_brief(execution),
        "memory_excerpt": (context.memory_long_term[:600] if context else ""),
        "retrieved_excerpt": _retrieval_excerpt(context),
        "task_results": [item.to_dict() for item in execution.task_results],
    }
    prompt = f"""# 角色
你是 MyFitness 的上下文反思模块。在生成最终回复前，核查用户个体事实是否已从**数据库查询结果**确认。

# 原则
- 若回答需要「当前/最新体重」，必须有 query_results.body.latest_metrics.weight 或身体 Agent 给出的最新体重。
- 用户画像、语义检索中的「初始体重」只能作为历史起点，不能替代当前体重。
- 若需要当前体重但数据库未检索到，ready=false，并说明应补做 body 数据查询。
- 若问题纯属通用知识、不涉及个体身体数字，ready=true。

# 输入
{json.dumps(payload, ensure_ascii=False, indent=2)}

# 输出 JSON（禁止其它文字）
{{
  "ready": true,
  "feedback": "<未就绪时说明缺什么>",
  "missing_fetches": ["body_latest"],
  "confirmed_notes": "<已确认、可写入回复的个体事实，一行中文>"
}}
"""
    try:
        content = chat_completion([{"role": "user", "content": prompt}], temperature=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("上下文反思 LLM 失败: %s", exc)
        return None

    data = _extract_json(content)
    if not isinstance(data, dict):
        return None

    ready = bool(data.get("ready"))
    retry_ids = list(fetch_task_ids)
    if not ready and not retry_ids:
        retry_ids = [
            result.task_id
            for result in execution.task_results
            if result.intent.value == "data_query"
        ]

    return ContextReflection(
        ready=ready,
        feedback=str(data.get("feedback") or ""),
        missing_fetches=[str(item) for item in data.get("missing_fetches") or []],
        confirmed_notes=str(data.get("confirmed_notes") or ""),
        retry_task_ids=retry_ids if not ready else [],
    )


def _agent_brief(execution: ExecutionResult) -> dict[str, str]:
    brief: dict[str, str] = {}
    if execution.agent_outputs.body and execution.agent_outputs.body.narrative:
        brief["body"] = execution.agent_outputs.body.narrative[:400]
    if execution.agent_outputs.nutrition and execution.agent_outputs.nutrition.narrative:
        brief["nutrition"] = execution.agent_outputs.nutrition.narrative[:400]
    return brief


def _retrieval_excerpt(context) -> str:
    if context is None or not context.retrieved_chunks:
        return ""
    parts: list[str] = []
    for item in context.retrieved_chunks[:3]:
        parts.append(str(item.get("content", ""))[:200])
    return "\n---\n".join(parts)


def _extract_json(content: str) -> dict | None:
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
