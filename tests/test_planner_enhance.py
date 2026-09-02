"""Planner 增强逻辑测试。"""

from datetime import date

from myfitness.graph.planner import build_task_plan
from myfitness.graph.planner_enhance import enhance_task_plan, infer_muscle_group, needs_training_history_fetch
from myfitness.graph.task_plan import PlannedTask, TaskPlan
from myfitness.schemas.state import Intent, RouteResult


def test_needs_training_history_fetch_for_back_day_plan():
    message = "按照安排，今天是练背日，根据我过往的训练记录，帮我安排一下今天的计划"
    assert needs_training_history_fetch(message)
    assert infer_muscle_group(message) == "背"


def test_enhance_adds_training_and_body_fetch_tasks():
    message = "今天是练背日，根据过往训练记录安排今天的训练"
    plan = TaskPlan(
        tasks=[
            PlannedTask(
                id="t1",
                intent=Intent.GENERAL,
                description="生成练背计划",
                domain="fitness",
            )
        ],
        user_requirements=message,
        primary_intent=Intent.GENERAL,
    )
    enhanced = enhance_task_plan(message, plan, date(2026, 9, 2))
    intents = {task.intent for task in enhanced.tasks}
    assert Intent.DATA_QUERY in intents
    training_fetch = next(
        task for task in enhanced.tasks if task.params.get("include_training_history")
    )
    assert training_fetch.start_date == date(2026, 8, 4)
    assert training_fetch.end_date == date(2026, 9, 2)
    assert training_fetch.params.get("muscle_group") == "背"
    general = next(task for task in enhanced.tasks if task.intent == Intent.GENERAL)
    assert training_fetch.id in general.depends_on


def test_build_task_plan_back_day_has_training_fetch():
    route = RouteResult(Intent.GENERAL)
    message = "按照安排，今天是练背日，根据我过往的训练记录，帮我安排一下今天的计划"
    plan = build_task_plan(message, route, today=date(2026, 9, 2), use_llm=False)
    assert any(task.params.get("include_training_history") for task in plan.tasks)
