"""手动录入、Planner、Judge、LangGraph 分析子图测试。"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END, START, StateGraph
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.manual_parser import parse_body_entry, parse_goal_weight
from myfitness.db.models import Base, User
from myfitness.graph.chat import new_chat_state
from myfitness.graph.context_reflection import ContextReflection
from myfitness.graph.judge import judge_turn
from myfitness.graph.langgraph_flow import (
    AnalysisGraphState,
    build_analysis_graph,
    get_analysis_graph,
    route_after_execute,
    route_after_judge,
    run_analysis_graph,
)
from myfitness.graph import langgraph_flow as flow
from myfitness.graph.planner import build_task_plan, should_use_orchestrator
from myfitness.graph.router import classify_intent
from myfitness.graph.task_plan import (
    ExecutionResult,
    JudgeVerdict,
    PlannedTask,
    TaskPlan,
    TaskResult,
)
from myfitness.memory.types import MemoryBundle
from myfitness.schemas.agent_outputs import SummaryAgentOutput
from myfitness.schemas.state import Intent, RouteResult


USER_MESSAGE = (
    "以2025年9月1日为起点，记录我的初始体重为130kg，"
    "初始体脂率为37%，目标是减到85千克，评价一下我减肥到今天的进度怎么样"
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    yield session
    session.close()


def _summary(text: str = "ok") -> SummaryAgentOutput:
    return SummaryAgentOutput(output_type="chat_reply", content_md=text)


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


def test_analysis_graph_has_expected_nodes():
    graph = build_analysis_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"plan", "execute_ready", "reflect", "judge", "summary"} <= nodes
    assert get_analysis_graph() is not None


def test_route_after_execute_ends_on_confirmation():
    assert route_after_execute({"needs_confirmation": True}) == "__end__"
    assert route_after_execute({"needs_confirmation": False}) == "reflect"


def test_route_after_judge_retries_then_summarizes():
    assert route_after_judge({"judge_satisfied": True, "judge_attempt": 1}) == "summary"
    assert (
        route_after_judge(
            {"judge_satisfied": False, "judge_attempt": 1, "retry_task_ids": ["t1"]}
        )
        == "execute_ready"
    )
    assert (
        route_after_judge(
            {"judge_satisfied": False, "judge_attempt": 99, "retry_task_ids": ["t1"]}
        )
        == "summary"
    )


def test_run_analysis_graph_happy_path(db_session):
    state = new_chat_state(1)
    state.user_message = "最近7天体重趋势"
    route = RouteResult(intents=[Intent.TREND_ANALYSIS], domain="body")
    plan = TaskPlan(
        tasks=[
            PlannedTask(
                id="t1",
                intent=Intent.TREND_ANALYSIS,
                description="分析体重趋势",
                domain="body",
            )
        ],
        primary_intent=Intent.TREND_ANALYSIS,
        domain="body",
    )
    progress: list = []

    with (
        patch("myfitness.graph.orchestrator.load_context_for_turn") as load_ctx,
        patch("myfitness.graph.orchestrator.run_body_agent") as body_agent,
        patch("myfitness.graph.langgraph_flow.reflect_before_answer") as reflect,
        patch("myfitness.graph.langgraph_flow.judge_turn") as judge,
        patch("myfitness.graph.langgraph_flow.run_summary_agent") as summary,
        patch("myfitness.graph.langgraph_flow.should_stream_summary", return_value=False),
    ):
        load_ctx.return_value = (MagicMock(), ["query_body_metrics"])
        body_agent.return_value = MagicMock()
        reflect.return_value = ContextReflection(ready=True, confirmed_notes="已确认体重")
        judge.return_value = JudgeVerdict(satisfied=True)
        summary.return_value = _summary("趋势平稳")

        execution, stream = run_analysis_graph(
            db_session,
            state,
            route,
            MemoryBundle(),
            on_progress=progress.append,
            plan=plan,
            use_llm=False,
        )

    assert stream is False
    assert execution.needs_confirmation is False
    assert execution.agent_outputs.summary is not None
    assert "summary" in execution.agents_invoked
    assert any(isinstance(item, dict) and item.get("type") == "task_plan" for item in progress)


def test_run_analysis_graph_judge_retry_reenters_execute(db_session):
    state = new_chat_state(1)
    state.user_message = "最近体重怎么样"
    route = RouteResult(intents=[Intent.DATA_QUERY], domain="body")
    plan = TaskPlan(
        tasks=[
            PlannedTask(
                id="t1",
                intent=Intent.DATA_QUERY,
                description="查体重",
                domain="body",
            )
        ],
        primary_intent=Intent.DATA_QUERY,
        domain="body",
    )
    calls = {"execute": 0, "judge": 0}

    def counting_execute(state_in, config=None):
        calls["execute"] += 1
        return flow.execute_ready_node(state_in, config)

    def judge_side_effect(*_args, **_kwargs):
        calls["judge"] += 1
        if calls["judge"] == 1:
            return JudgeVerdict(satisfied=False, feedback="再查一次", retry_task_ids=["t1"])
        return JudgeVerdict(satisfied=True)

    with (
        patch("myfitness.graph.orchestrator.load_context_for_turn", return_value=(MagicMock(), [])),
        patch("myfitness.graph.orchestrator.run_body_agent", return_value=MagicMock()),
        patch(
            "myfitness.graph.langgraph_flow.reflect_before_answer",
            return_value=ContextReflection(ready=True),
        ),
        patch("myfitness.graph.langgraph_flow.judge_turn", side_effect=judge_side_effect),
        patch(
            "myfitness.graph.langgraph_flow.run_summary_agent",
            return_value=_summary("ok"),
        ),
        patch("myfitness.graph.langgraph_flow.should_stream_summary", return_value=False),
    ):
        graph = StateGraph(AnalysisGraphState)
        graph.add_node("plan", flow.plan_node)
        graph.add_node("execute_ready", counting_execute)
        graph.add_node("reflect", flow.reflect_node)
        graph.add_node("judge", flow.judge_node)
        graph.add_node("summary", flow.summary_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute_ready")
        graph.add_conditional_edges(
            "execute_ready",
            flow.route_after_execute,
            {"reflect": "reflect", "__end__": END},
        )
        graph.add_edge("reflect", "judge")
        graph.add_conditional_edges(
            "judge",
            flow.route_after_judge,
            {"execute_ready": "execute_ready", "summary": "summary"},
        )
        graph.add_edge("summary", END)
        compiled = graph.compile()
        previous = flow._ANALYSIS_GRAPH
        flow._ANALYSIS_GRAPH = compiled
        try:
            execution, _stream = run_analysis_graph(
                db_session,
                state,
                route,
                MemoryBundle(),
                plan=plan,
                use_llm=False,
            )
        finally:
            flow._ANALYSIS_GRAPH = previous

    assert calls["execute"] >= 2
    assert calls["judge"] >= 2
    assert execution.agent_outputs.summary is not None


def test_run_analysis_graph_stops_on_pending_confirmation(db_session):
    state = new_chat_state(1)
    state.user_message = "记录体重 73kg"
    route = RouteResult(intents=[Intent.MANUAL_ENTRY], domain="body")
    plan = TaskPlan(
        tasks=[
            PlannedTask(
                id="t1",
                intent=Intent.MANUAL_ENTRY,
                description="录入体重",
                domain="body",
            )
        ],
        primary_intent=Intent.MANUAL_ENTRY,
        domain="body",
    )

    with patch("myfitness.graph.langgraph_flow.run_summary_agent") as summary:
        execution, stream = run_analysis_graph(
            db_session,
            state,
            route,
            MemoryBundle(),
            plan=plan,
            use_llm=False,
        )

    assert stream is False
    assert execution.needs_confirmation is True
    assert state.pending_confirmation is not None
    summary.assert_not_called()
