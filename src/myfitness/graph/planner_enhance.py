"""Planner 后处理 — 补齐数据检索任务、日期范围与依赖关系。"""

from __future__ import annotations

import re
from datetime import date, timedelta

from myfitness.graph.context_reflection import needs_personalized_context
from myfitness.graph.task_plan import PlannedTask, TaskPlan
from myfitness.schemas.state import Intent

_TRAINING_PLAN_RE = re.compile(
    r"(练背|练胸|练腿|练肩|练手臂|训练计划|安排.*训练|今天.*练|怎么练|练什么|"
    r"背部训练|胸部训练|腿部训练)"
)
_HISTORY_HINT_RE = re.compile(
    r"(过往|历史|上次|以往|以前|结合|根据).*(训练|练|记录)|"
    r"训练记录|练.*记录|记录.*练"
)
_MUSCLE_HINTS = (
    ("练背", "背"),
    ("背部", "背"),
    ("背日", "背"),
    ("练胸", "胸"),
    ("胸部", "胸"),
    ("练腿", "腿"),
    ("腿部", "腿"),
    ("练肩", "肩"),
    ("二头", "二头"),
    ("三头", "三头"),
)

_ANALYSIS_INTENTS = {
    Intent.DATA_QUERY,
    Intent.TREND_ANALYSIS,
    Intent.WEB_SEARCH,
    Intent.GENERAL,
    Intent.PLAN_ADJUST,
}


def enhance_task_plan(message: str, plan: TaskPlan, today: date) -> TaskPlan:
    """在 LLM/规则计划之上补齐必要的检索子任务与依赖。"""
    tasks = list(plan.tasks)
    if not tasks:
        return plan

    body_fetch_id = _find_task_id(
        tasks, lambda t: t.params.get("include_latest_body") or t.params.get("scope") == "confirm"
    )
    training_fetch_id = _find_task_id(
        tasks,
        lambda t: t.params.get("include_training_history")
        or t.params.get("scope") == "training_history",
    )

    next_index = _next_task_index(tasks)

    if needs_training_history_fetch(message) and training_fetch_id is None:
        muscle = infer_muscle_group(message)
        training_fetch_id = f"t{next_index}"
        next_index += 1
        label = f"检索近30天训练历史（{muscle}）" if muscle else "检索近30天训练历史"
        tasks.append(
            PlannedTask(
                id=training_fetch_id,
                intent=Intent.DATA_QUERY,
                description=label,
                domain="fitness",
                start_date=today - timedelta(days=29),
                end_date=today,
                depends_on=[],
                params={
                    "scope": "training_history",
                    "include_training_history": True,
                    "muscle_group": muscle,
                },
            )
        )

    if needs_personalized_context(message) and body_fetch_id is None:
        body_fetch_id = f"t{next_index}"
        next_index += 1
        tasks.append(
            PlannedTask(
                id=body_fetch_id,
                intent=Intent.DATA_QUERY,
                description="检索并确认最新身体指标",
                domain="body",
                depends_on=[],
                params={"scope": "confirm", "include_latest_body": True},
            )
        )

    fetch_deps = [item for item in (body_fetch_id, training_fetch_id) if item]
    if fetch_deps:
        tasks = _wire_fetch_dependencies(tasks, fetch_deps)

    tasks = _sort_tasks_by_dependency(tasks)
    primary = _resolve_primary_intent(tasks, plan.primary_intent)
    return TaskPlan(
        tasks=tasks,
        user_requirements=plan.user_requirements or message.strip(),
        primary_intent=primary,
        domain=plan.domain or _infer_primary_domain(tasks),
        start_date=plan.start_date,
        end_date=plan.end_date or today,
    )


def needs_training_history_fetch(message: str) -> bool:
    if _TRAINING_PLAN_RE.search(message):
        return True
    if _HISTORY_HINT_RE.search(message) and re.search(r"(训练|练)", message):
        return True
    return False


def infer_muscle_group(message: str) -> str | None:
    for hint, muscle in _MUSCLE_HINTS:
        if hint in message:
            return muscle
    if "背" in message and re.search(r"练|训练", message):
        return "背"
    return None


def _find_task_id(tasks: list[PlannedTask], predicate) -> str | None:
    for task in tasks:
        if predicate(task):
            return task.id
    return None


def _next_task_index(tasks: list[PlannedTask]) -> int:
    indices: list[int] = []
    for task in tasks:
        if task.id.startswith("t") and task.id[1:].isdigit():
            indices.append(int(task.id[1:]))
    return max(indices, default=0) + 1


def _wire_fetch_dependencies(
    tasks: list[PlannedTask],
    fetch_deps: list[str],
) -> list[PlannedTask]:
    fetch_set = set(fetch_deps)
    updated: list[PlannedTask] = []
    for task in tasks:
        if task.id in fetch_set:
            updated.append(task)
            continue
        if task.intent in _ANALYSIS_INTENTS or task.intent == Intent.GOAL_SETTING:
            merged = list(dict.fromkeys([*task.depends_on, *fetch_deps]))
            task.depends_on = merged
        updated.append(task)
    return updated


def _sort_tasks_by_dependency(tasks: list[PlannedTask]) -> list[PlannedTask]:
    """把无依赖的检索任务排到前面，便于前端展示与执行。"""
    fetch_tasks = [t for t in tasks if not t.depends_on and t.intent == Intent.DATA_QUERY]
    others = [t for t in tasks if t not in fetch_tasks]
    return fetch_tasks + others


def _resolve_primary_intent(tasks: list[PlannedTask], fallback: Intent) -> Intent:
    for task in reversed(tasks):
        if task.intent in _ANALYSIS_INTENTS:
            return task.intent
    return fallback


def _infer_primary_domain(tasks: list[PlannedTask]) -> str | None:
    for task in reversed(tasks):
        if task.domain:
            return task.domain
    return None
