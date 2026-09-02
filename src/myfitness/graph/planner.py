"""Planner — 将用户消息拆分为可执行任务（含依赖关系）。"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from myfitness.agents.intent_agent import _extract_json, _parse_date_range, _parse_domain
from myfitness.graph.context_reflection import needs_personalized_context
from myfitness.debug import trace_agent
from myfitness.graph.planner_enhance import enhance_task_plan
from myfitness.graph.task_plan import PlannedTask, TaskPlan
from myfitness.llm.factory import chat_completion, is_llm_configured
from myfitness.schemas.state import Intent, RouteResult

logger = logging.getLogger(__name__)

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_MAX_TASKS = 6

_ANALYSIS_INTENTS = {
    Intent.DATA_QUERY,
    Intent.TREND_ANALYSIS,
    Intent.WEB_SEARCH,
    Intent.GENERAL,
}


def build_task_plan(
    message: str,
    route: RouteResult,
    *,
    today: date | None = None,
    use_llm: bool | None = None,
) -> TaskPlan:
    """根据用户消息与路由结果生成任务计划。"""
    today = today or datetime.now(_LOCAL_TZ).date()
    llm_enabled = is_llm_configured() if use_llm is None else use_llm
    if llm_enabled:
        llm_plan = _llm_plan(message, route, today)
        if llm_plan is not None:
            return enhance_task_plan(message, llm_plan, today)
    plan = _rule_plan(message, route, today, use_llm=llm_enabled)
    return enhance_task_plan(message, plan, today)


def should_use_orchestrator(
    route: RouteResult,
    message: str,
    *,
    use_llm: bool | None = None,
) -> bool:
    """是否走 Planner + Orchestrator + Judge 路径。"""
    llm_enabled = is_llm_configured() if use_llm is None else use_llm
    if len(route.intents) > 1:
        return True
    if route.has(Intent.MANUAL_ENTRY) and any(
        route.has(item) for item in (Intent.TREND_ANALYSIS, Intent.GOAL_SETTING, Intent.DATA_QUERY)
    ):
        return True
    if _is_compound_request(message):
        return True
    if route.intent in _ANALYSIS_INTENTS:
        return True
    if route.has(Intent.PLAN_ADJUST):
        return True
    if route.intent == Intent.GENERAL and llm_enabled:
        return True
    return False


def _is_compound_request(message: str) -> bool:
    has_manual = bool(re.search(r"(记录|录入|初始).*(体重|体脂)", message))
    has_followup = bool(
        re.search(r"(评价|进度|怎么样|分析|趋势|变化|目标|减到|降到|增到)", message)
    )
    return has_manual and has_followup


@trace_agent("Planner")
def _llm_plan(message: str, route: RouteResult, today: date) -> TaskPlan | None:
    if not is_llm_configured():
        return None
    try:
        content = chat_completion(
            [
                {"role": "system", "content": _build_planner_prompt(today)},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Planner LLM 调用失败: %s", exc)
        return None
    return _parse_planner_response(content, route, today)


def _build_planner_prompt(today: date) -> str:
    return f"""# 角色
你是 MyFitness 的任务 Planner。根据用户消息拆分为多个子任务，并标注依赖关系。

# 当前日期
{today.isoformat()}

# 任务类型（task_type）
- manual_entry: 手动录入体重/体脂/饮食（写入前需用户确认）
- goal_setting: 设定体重/体脂目标
- sync_trigger / report_trigger / chart_trigger / schedule_manage: 动作类
- data_query: 查询已有记录
- trend_analysis: 趋势/进度/评价类分析
- web_search: 联网检索公开资料
- general: 寒暄或说明

# 规则
1. 一条消息含「记录初始数据 + 设定目标 + 评价进度」时，必须拆成多个任务，不能合并。
2. 录入类任务的 params 中填写解析出的 record_date、weight_kg、bodyfat_pct 等（如有）。
3. 评价/进度/趋势类任务通常 depends_on 录入任务（若同轮需先写入基准数据）。
4. 无依赖关系的任务 depends_on 留空数组，Orchestrator 将并行执行。
5. 数字必须绑定到正确字段：年份不是体重，「130kg」才是体重，「37%」才是体脂。
6. 若用户要求个性化饮食/减脂/训练建议，须先增加 data_query 任务（domain=body，params 含 include_latest_body=true）检索并确认最新身体数据，再 depends_on 该任务执行分析/回答。
7. 若用户要「今天练 xxx /安排训练计划/根据过往记录」，须先增加 data_query（domain=fitness，params 含 include_training_history=true，date_range 近30天），再 depends_on 执行分析与计划生成；「今天」是安排目标日，训练历史必须查近30天，不能只查今天。
8. 检索类任务与回答类任务必须拆分；回答任务 depends_on 所有检索任务。
9. 最多 {_MAX_TASKS} 个任务。

# 输出 JSON（禁止其它文字）
{{
  "user_requirements": "<一句话概括用户要什么>",
  "tasks": [
    {{
      "id": "t1",
      "task_type": "manual_entry",
      "description": "记录2025-09-01初始体重130kg体脂37%",
      "domain": "body",
      "date_range": {{"start": "2025-09-01", "end": "2025-09-01"}},
      "depends_on": [],
      "params": {{"record_date": "2025-09-01", "weight_kg": 130, "bodyfat_pct": 37}}
    }},
    {{
      "id": "t2",
      "task_type": "goal_setting",
      "description": "设定目标体重85kg",
      "domain": "body",
      "date_range": null,
      "depends_on": [],
      "params": {{"target_weight_kg": 85}}
    }},
    {{
      "id": "t3",
      "task_type": "trend_analysis",
      "description": "评价从起点至今的减肥进度",
      "domain": "body",
      "date_range": {{"start": "2025-09-01", "end": "{today.isoformat()}"}},
      "depends_on": ["t1"],
      "params": {{}}
    }}
  ]
}}
"""


def _parse_planner_response(content: str, route: RouteResult, today: date) -> TaskPlan | None:
    data = _extract_json(content)
    if not isinstance(data, dict):
        return None
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return None

    tasks: list[PlannedTask] = []
    for index, item in enumerate(raw_tasks[:_MAX_TASKS]):
        if not isinstance(item, dict):
            continue
        task_type = str(item.get("task_type") or item.get("intent") or "").strip().lower()
        if task_type not in {intent.value for intent in Intent}:
            continue
        start_date, end_date = _parse_date_range(item.get("date_range"), today)
        task_id = str(item.get("id") or f"t{index + 1}")
        tasks.append(
            PlannedTask(
                id=task_id,
                intent=Intent(task_type),
                description=str(item.get("description") or ""),
                domain=_parse_domain(item.get("domain")) or route.domain,
                start_date=start_date,
                end_date=end_date,
                depends_on=[str(dep) for dep in item.get("depends_on") or []],
                params=dict(item.get("params") or {}),
            )
        )

    if not tasks:
        return None

    primary = tasks[-1].intent if tasks[-1].intent in _ANALYSIS_INTENTS else tasks[0].intent
    return TaskPlan(
        tasks=tasks,
        user_requirements=str(data.get("user_requirements") or ""),
        primary_intent=primary,
        domain=route.domain or tasks[0].domain,
        start_date=route.start_date or tasks[0].start_date,
        end_date=route.end_date or tasks[-1].end_date or today,
    )


def _rule_plan(
    message: str,
    route: RouteResult,
    today: date,
    *,
    use_llm: bool = False,
) -> TaskPlan:
    """LLM 不可用时的规则拆分。"""
    tasks: list[PlannedTask] = []
    task_index = 1

    def _next_id() -> str:
        nonlocal task_index
        task_id = f"t{task_index}"
        task_index += 1
        return task_id

    manual_id: str | None = None
    if route.has(Intent.MANUAL_ENTRY) or re.search(r"(记录|录入|初始).*(体重|体脂)", message):
        manual_id = _next_id()
        start = route.start_date
        tasks.append(
            PlannedTask(
                id=manual_id,
                intent=Intent.MANUAL_ENTRY,
                description="手动录入身体/饮食数据",
                domain=route.domain or "body",
                start_date=start,
                end_date=start or route.end_date,
                depends_on=[],
            )
        )

    if route.has(Intent.GOAL_SETTING) or re.search(
        r"(目标|减到|降到|增到).*(kg|公斤|千克|%)", message, re.I
    ):
        tasks.append(
            PlannedTask(
                id=_next_id(),
                intent=Intent.GOAL_SETTING,
                description="设定身体目标",
                domain="body",
                depends_on=[manual_id] if manual_id else [],
            )
        )

    fetch_id: str | None = None
    if needs_personalized_context(message):
        fetch_id = _next_id()
        tasks.append(
            PlannedTask(
                id=fetch_id,
                intent=Intent.DATA_QUERY,
                description="检索并确认用户最新身体数据",
                domain="body",
                params={"scope": "confirm", "include_latest_body": True},
                depends_on=[],
            )
        )

    analysis_intent = Intent.TREND_ANALYSIS
    if route.has(Intent.DATA_QUERY):
        analysis_intent = Intent.DATA_QUERY
    elif route.has(Intent.TREND_ANALYSIS):
        analysis_intent = Intent.TREND_ANALYSIS
    elif route.has(Intent.WEB_SEARCH):
        analysis_intent = Intent.WEB_SEARCH
    elif route.has(Intent.GENERAL):
        analysis_intent = Intent.GENERAL

    needs_analysis = (
        route.has(Intent.TREND_ANALYSIS)
        or route.has(Intent.DATA_QUERY)
        or route.has(Intent.WEB_SEARCH)
        or (route.has(Intent.GENERAL) and (use_llm or needs_personalized_context(message)))
        or re.search(r"(评价|进度|怎么样|分析|趋势|变化)", message)
    )
    if needs_analysis and analysis_intent not in {Intent.MANUAL_ENTRY}:
        depends = [dep for dep in (manual_id, fetch_id) if dep]
        tasks.append(
            PlannedTask(
                id=_next_id(),
                intent=analysis_intent,
                description="分析或评价用户问题",
                domain=route.domain,
                start_date=route.start_date,
                end_date=route.end_date or today,
                depends_on=depends,
            )
        )

    if not tasks:
        for intent in route.intents:
            tasks.append(
                PlannedTask(
                    id=_next_id(),
                    intent=intent,
                    description=intent.value,
                    domain=route.domain,
                    start_date=route.start_date,
                    end_date=route.end_date,
                )
            )

    primary = next(
        (task.intent for task in reversed(tasks) if task.intent in _ANALYSIS_INTENTS),
        tasks[0].intent,
    )
    return TaskPlan(
        tasks=tasks,
        user_requirements=message.strip(),
        primary_intent=primary,
        domain=route.domain,
        start_date=route.start_date,
        end_date=route.end_date or today,
    )
