"""手动录入、Planner、Judge 测试。"""

from datetime import date

from myfitness.agents.manual_parser import parse_body_entry, parse_goal_weight
from myfitness.graph.judge import judge_turn
from myfitness.graph.planner import build_task_plan, should_use_orchestrator
from myfitness.graph.router import classify_intent
from myfitness.graph.task_plan import ExecutionResult, TaskPlan, TaskResult
from myfitness.schemas.state import Intent, RouteResult


USER_MESSAGE = (
    "以2025年9月1日为起点，记录我的初始体重为130kg，"
    "初始体脂率为37%，目标是减到85千克，评价一下我减肥到今天的进度怎么样"
)


def test_parse_body_entry_ignores_year_before_weight():
    payload = parse_body_entry(USER_MESSAGE, date(2025, 9, 1))
    assert payload is not None
    records = {item["metric_type"]: item for item in payload["records"]}
    assert records["weight"]["value"] == 130.0
    assert records["bodyfat"]["value"] == 37.0
    assert records["weight"]["record_date"] == "2025-09-01"


def test_parse_body_entry_rejects_bare_year_as_weight():
    payload = parse_body_entry("2025年体重72.5kg")
    assert payload is not None
    assert payload["records"][0]["value"] == 72.5


def test_parse_goal_weight_from_message():
    assert parse_goal_weight(USER_MESSAGE) == 85.0


def test_router_compound_manual_goal_trend():
    route = classify_intent(USER_MESSAGE, use_llm=False, today=date(2026, 8, 31))
    assert route.has(Intent.MANUAL_ENTRY)
    assert route.has(Intent.GOAL_SETTING)
    assert route.has(Intent.TREND_ANALYSIS)


def test_should_use_orchestrator_for_compound_message():
    route = RouteResult(
        intents=[Intent.MANUAL_ENTRY, Intent.GOAL_SETTING, Intent.TREND_ANALYSIS],
        domain="body",
    )
    assert should_use_orchestrator(route, USER_MESSAGE)


def test_rule_planner_splits_compound_message():
    route = classify_intent(USER_MESSAGE, use_llm=False, today=date(2026, 8, 31))
    plan = build_task_plan(USER_MESSAGE, route, today=date(2026, 8, 31))
    intents = [task.intent for task in plan.tasks]
    assert Intent.MANUAL_ENTRY in intents
    assert Intent.GOAL_SETTING in intents
    assert Intent.TREND_ANALYSIS in intents


def test_judge_rejects_confirmation_without_analysis():
    plan = TaskPlan(
        tasks=[],
        user_requirements=USER_MESSAGE,
        primary_intent=Intent.TREND_ANALYSIS,
    )
    execution = ExecutionResult(
        needs_confirmation=True,
        task_results=[
            TaskResult(
                task_id="t1",
                intent=Intent.MANUAL_ENTRY,
                status="pending_confirmation",
                summary="请确认写入",
            )
        ],
    )
    verdict = judge_turn(USER_MESSAGE, plan, execution)
    assert verdict.satisfied is False
    assert "进度" in verdict.feedback or "确认" in verdict.feedback
