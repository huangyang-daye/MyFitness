"""首响性能相关测试。"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.tools.query_planner import QueryPlan, build_query_plan
from myfitness.db.models import Base, BodyMetric, User
from myfitness.db.repositories.metrics import SOURCE_MANUAL
from myfitness.graph.chat import _agents_for_turn, new_chat_state, prepare_chat_turn
from myfitness.graph.router import RouteResult, classify_intent
from myfitness.schemas.state import Intent
from myfitness.services.context_loader import load_context_snapshot


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, name="test")
    session.add(user)
    session.flush()
    yield session
    session.close()


def test_agents_for_turn_only_body_on_weight_query():
    route = RouteResult(Intent.TREND_ANALYSIS)
    plan = build_query_plan("最近7天的体重", Intent.TREND_ANALYSIS, today=date(2026, 8, 23))
    assert plan is not None
    assert _agents_for_turn(route, plan) == ["body"]


def test_classify_recent_weight_as_trend_or_data():
    result = classify_intent("最近7天的体重")
    assert result.intent in {Intent.TREND_ANALYSIS, Intent.DATA_QUERY}


def test_load_context_reuses_query_results_without_extra_body_query(db_session):
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 23),
            metric_type="weight",
            value=71.0,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    query_results = {
        "body": {
            "tool": "query_body_metrics",
            "start_date": "2026-08-17",
            "end_date": "2026-08-23",
            "count": 1,
            "records": [
                {
                    "date": "2026-08-23",
                    "metric_type": "weight",
                    "value": 71.0,
                    "unit": "kg",
                    "source": "manual",
                }
            ],
        }
    }

    with patch.object(
        __import__("myfitness.services.context_loader", fromlist=["BodyMetricRepository"]).BodyMetricRepository,
        "query_range",
        side_effect=AssertionError("不应重复查询 body"),
    ):
        ctx = load_context_snapshot(
            db_session,
            1,
            end_date=date(2026, 8, 23),
            lookback_days=7,
            query_results=query_results,
        )

    assert ctx.body_metrics_summary["latest_weight_kg"] == 71.0


def test_prepare_chat_turn_runs_single_agent_for_weight(db_session):
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 23),
            metric_type="weight",
            value=71.0,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        result = prepare_chat_turn(db_session, state, "最近7天的体重")

    assert result.state.metadata.agents_invoked == ["body_monitor", "summary"]
    assert result.state.agent_outputs.nutrition is None
    assert result.state.agent_outputs.fitness is None
