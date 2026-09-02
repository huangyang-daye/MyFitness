"""上下文反思与个性化调度测试。"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.tools.query_planner import build_query_plan
from myfitness.db.models import Base, BodyMetric, User
from myfitness.graph.context_reflection import (
    needs_personalized_context,
    reflect_before_answer,
)
from myfitness.graph.planner import build_task_plan
from myfitness.graph.task_plan import ExecutionResult
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import ContextSnapshot, DateRange, Intent, RouteResult


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(User(id=1, name="u1"))
    session.flush()
    yield session
    session.close()


def test_needs_personalized_context_for_diet_advice():
    message = "根据中国居民膳食指南，我该如何安排减脂期间的饮食"
    assert needs_personalized_context(message)


def test_build_query_plan_adds_latest_body_for_personalized_general():
    plan = build_query_plan(message="减脂期间如何安排饮食", intent=Intent.GENERAL)
    assert plan is not None
    assert plan.include_latest_body is True
    assert "body" in plan.domains


def test_rule_plan_inserts_confirm_fetch_before_analysis():
    route = RouteResult(Intent.GENERAL)
    plan = build_task_plan(
        "根据中国居民膳食指南，我该如何安排减脂期间的饮食",
        route,
        use_llm=False,
    )
    fetch_tasks = [
        task for task in plan.tasks if task.params.get("include_latest_body")
    ]
    assert fetch_tasks
    analysis = next(task for task in plan.tasks if task.intent == Intent.GENERAL)
    assert fetch_tasks[0].id in analysis.depends_on


def test_reflect_blocks_when_latest_weight_missing():
    execution = ExecutionResult(
        context=ContextSnapshot(
            date_range=DateRange(start=date(2026, 8, 26), end=date(2026, 9, 1)),
            memory_long_term="初始体重130kg",
        ),
        agent_outputs=AgentOutputs(),
    )
    result = reflect_before_answer(
        "减脂期间如何安排饮食，结合我的体重",
        execution,
        use_llm=False,
    )
    assert result.ready is False
    assert "latest" in " ".join(result.missing_fetches).lower() or "body" in result.feedback


def test_reflect_passes_when_latest_weight_in_query(db_session):
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 28),
            metric_type="weight",
            value=118,
            unit="kg",
            source="manual",
        )
    )
    db_session.flush()

    execution = ExecutionResult(
        context=ContextSnapshot(
            date_range=DateRange(start=date(2026, 8, 26), end=date(2026, 9, 1)),
            query_results={
                "body": {
                    "latest_metrics": {
                        "weight": {
                            "value": 118.0,
                            "unit": "kg",
                            "date": "2026-08-28",
                            "source": "manual",
                        }
                    }
                }
            },
        ),
        agent_outputs=AgentOutputs(),
    )
    result = reflect_before_answer(
        "减脂期间如何安排饮食，结合我的体重",
        execution,
        use_llm=False,
    )
    assert result.ready is True
    assert "118" in result.confirmed_notes
