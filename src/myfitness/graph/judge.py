"""Judge — 直接对照用户原始需求评估任务产出是否满足要求。"""

from __future__ import annotations

import json
import logging
import re

from myfitness.debug import trace_agent
from myfitness.graph.context_reflection import needs_body_metrics_confirmation
from myfitness.graph.task_plan import ExecutionResult, JudgeVerdict, TaskPlan
from myfitness.llm.factory import chat_completion, is_llm_configured

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


def max_judge_attempts() -> int:
    return _MAX_RETRIES + 1


@trace_agent("Judge")
def judge_turn(
    user_message: str,
    plan: TaskPlan,
    execution: ExecutionResult,
    *,
    attempt: int = 1,
    use_llm: bool | None = None,
) -> JudgeVerdict:
    """Judge 直接分析用户输入与执行结果，不经过意图识别 Agent。"""
    llm_enabled = is_llm_configured() if use_llm is None else use_llm
    if llm_enabled:
        llm_verdict = _llm_judge(user_message, plan, execution, attempt=attempt)
        if llm_verdict is not None:
            return llm_verdict
    return _rule_judge(user_message, plan, execution)


def _llm_judge(
    user_message: str,
    plan: TaskPlan,
    execution: ExecutionResult,
    *,
    attempt: int,
) -> JudgeVerdict | None:
    payload = {
        "user_requirements": plan.user_requirements or user_message,
        "task_results": [item.to_dict() for item in execution.task_results],
        "agent_summaries": _agent_summary(execution),
        "errors": execution.errors,
        "needs_confirmation": execution.needs_confirmation,
        "attempt": attempt,
    }
    prompt = f"""# 角色
你是 MyFitness 的质量 Judge。请**直接阅读用户原始消息**与下方执行结果，判断是否已经满足用户的全部要求。

# 注意
- 不要调用或假设意图识别 Agent 的结论；以用户原文为准。
- 若用户同时要求「录入数据」和「评价进度」，仅完成录入确认而没有任何进度评价，则 satisfied=false。
- 若解析出的数值明显错误（如体重=2025kg），satisfied=false。
- 若仍在等待用户确认写入（needs_confirmation=true），且用户还要求分析/评价，则 satisfied=false。
- 若用户要求个性化饮食/减脂建议且回答会涉及体重，但执行结果中缺少数据库 latest_metrics 或身体 Agent 的最新体重，satisfied=false。
- 若用户要求保存/导出为 PDF、Word、Markdown 等文档，文件由系统在回复后自动落盘；只需检查分析/建议内容是否充分，不要因为尚未看到文件路径而判失败。
- 最多允许 {_MAX_RETRIES} 次重做；当前为第 {attempt} 次评估。

# 用户原始消息
{user_message}

# 执行结果
{json.dumps(payload, ensure_ascii=False, indent=2)}

# 输出 JSON（禁止其它文字）
{{
  "satisfied": true,
  "feedback": "<若不满足，说明缺什么/错在哪>",
  "missing": ["<未覆盖的需求点>"],
  "retry_task_ids": ["<建议重做的 task_id>"]
}}
"""
    try:
        content = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge LLM 调用失败: %s", exc)
        return None

    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    return JudgeVerdict(
        satisfied=bool(data.get("satisfied")),
        feedback=str(data.get("feedback") or ""),
        missing=[str(item) for item in data.get("missing") or []],
        retry_task_ids=[str(item) for item in data.get("retry_task_ids") or []],
    )


def _rule_judge(user_message: str, plan: TaskPlan, execution: ExecutionResult) -> JudgeVerdict:
    missing: list[str] = []

    if execution.needs_confirmation:
        if re.search(r"(评价|进度|怎么样|分析|趋势)", user_message):
            missing.append("尚未完成数据写入确认，无法进行进度评价")
        else:
            return JudgeVerdict(satisfied=True, feedback="等待用户确认写入")

    if re.search(r"(评价|进度|怎么样|分析|趋势|变化)", user_message):
        has_analysis = any(
            result.intent.value in {"trend_analysis", "data_query", "general", "web_search"}
            and result.status == "success"
            for result in execution.task_results
        )
        has_agent_output = bool(
            execution.agent_outputs.body
            or execution.agent_outputs.nutrition
            or execution.agent_outputs.fitness
        )
        if not has_analysis and not has_agent_output:
            missing.append("缺少进度/趋势分析结果")

    if re.search(r"(记录|录入|初始).*(体重|体脂)", user_message):
        manual = next(
            (item for item in execution.task_results if item.intent.value == "manual_entry"),
            None,
        )
        if manual and manual.status == "failed":
            missing.append("未能正确解析或准备手动录入")
        if any("2025.0" in item.summary for item in execution.task_results):
            missing.append("录入数值解析错误（疑似把年份当成体重/体脂）")

    if needs_body_metrics_confirmation(user_message):
        body = ((execution.context.query_results or {}).get("body") if execution.context else {}) or {}
        has_latest = bool((body.get("latest_metrics") or {}).get("weight"))
        has_body_agent = bool(
            execution.agent_outputs.body
            and execution.agent_outputs.body.current_metrics.weight_kg is not None
        )
        if not has_latest and not has_body_agent:
            missing.append("个性化建议需要先检索数据库最新身体指标")

    if missing:
        return JudgeVerdict(
            satisfied=False,
            feedback="；".join(missing),
            missing=missing,
        )
    if execution.errors:
        return JudgeVerdict(
            satisfied=False,
            feedback="；".join(execution.errors),
            missing=list(execution.errors),
        )
    return JudgeVerdict(satisfied=True)


def _agent_summary(execution: ExecutionResult) -> dict[str, str]:
    summaries: dict[str, str] = {}
    if execution.agent_outputs.body and execution.agent_outputs.body.narrative:
        summaries["body"] = execution.agent_outputs.body.narrative[:800]
    if execution.agent_outputs.nutrition and execution.agent_outputs.nutrition.narrative:
        summaries["nutrition"] = execution.agent_outputs.nutrition.narrative[:800]
    if execution.agent_outputs.fitness and execution.agent_outputs.fitness.narrative:
        summaries["fitness"] = execution.agent_outputs.fitness.narrative[:800]
    if execution.agent_outputs.summary and execution.agent_outputs.summary.content_md:
        summaries["summary"] = execution.agent_outputs.summary.content_md[:800]
    return summaries


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
