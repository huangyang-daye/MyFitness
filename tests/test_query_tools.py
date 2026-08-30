from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.query_planner import build_query_plan, parse_single_date
from myfitness.agents.tools.query_tools import query_body_metrics, query_nutrition_logs
from myfitness.db.models import Base, BodyMetric, NutritionLog, User
from myfitness.db.repositories.metrics import SOURCE_MANUAL
from myfitness.graph.chat import new_chat_state, prepare_chat_turn
from myfitness.schemas.state import Intent


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


def test_build_query_plan_yesterday_protein(db_session):
    plan = build_query_plan("昨天吃了多少蛋白质？", Intent.DATA_QUERY, today=date(2026, 8, 22))
    assert plan is not None
    assert plan.start_date == date(2026, 8, 21)
    assert plan.end_date == date(2026, 8, 21)
    assert "nutrition" in plan.domains


def test_parse_single_date_variants():
    today = date(2026, 8, 23)
    assert parse_single_date("生成今天日报", today) == today
    assert parse_single_date("生成昨天日报", today) == date(2026, 8, 22)
    assert parse_single_date("生成8.21的报告", today) == date(2026, 8, 21)
    assert parse_single_date("生成8月20日日报", today) == date(2026, 8, 20)
    assert parse_single_date("生成2026-08-19日报", today) == date(2026, 8, 19)
    assert parse_single_date("生成日报", today, default=date(2026, 8, 22)) == date(2026, 8, 22)


def test_build_query_plan_recent_7_days_includes_today():
    today = date(2026, 8, 23)
    plan = build_query_plan("最近7天的体重", Intent.DATA_QUERY, today=today)
    assert plan is not None
    assert plan.start_date == date(2026, 8, 17)
    assert plan.end_date == today
    assert plan.domains == ("body",)
    assert plan.metric_type == "weight"


def test_build_query_plan_default_range_includes_today():
    today = date(2026, 8, 23)
    plan = build_query_plan("体重多少", Intent.DATA_QUERY, today=today)
    assert plan is not None
    assert plan.end_date == today
    assert plan.lookback_days == 7


def test_build_query_plan_trend_analysis_includes_today():
    today = date(2026, 8, 23)
    plan = build_query_plan("近30天体脂变化趋势", Intent.TREND_ANALYSIS, today=today)
    assert plan is not None
    assert plan.start_date == date(2026, 7, 25)
    assert plan.end_date == today


def test_query_nutrition_logs(db_session):
    db_session.add(
        NutritionLog(
            user_id=1,
            record_date=date(2026, 8, 21),
            meal_type="lunch",
            food_name="鸡胸肉",
            amount=200,
            unit="g",
            nutrients_snapshot={"cal": 330, "protein": 62, "fat": 7, "carb": 0},
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    result = invoke_tool(
        query_nutrition_logs, db_session, 1, start_date=date(2026, 8, 21), end_date=date(2026, 8, 21)
    )
    assert result["count"] == 1
    assert result["daily_totals"]["2026-08-21"]["protein_g"] == 62


def test_query_body_metrics(db_session):
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 21),
            metric_type="weight",
            value=72.5,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    result = invoke_tool(
        query_body_metrics, db_session, 1, start_date=date(2026, 8, 21), end_date=date(2026, 8, 21)
    )
    assert result["count"] == 1
    assert result["records"][0]["value"] == 72.5


def test_prepare_chat_turn_invokes_query_tools(db_session):
    db_session.add(
        NutritionLog(
            user_id=1,
            record_date=date(2026, 8, 21),
            meal_type="lunch",
            food_name="鸡蛋",
            amount=2,
            unit="个",
            nutrients_snapshot={"cal": 144, "protein": 13, "fat": 10, "carb": 1},
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    state = new_chat_state(user_id=1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("myfitness.graph.chat.is_llm_configured", lambda: False)
        result = prepare_chat_turn(db_session, state, "昨天吃了多少蛋白质？", )

    assert "query_nutrition_logs" in result.state.metadata.tools_invoked
    assert result.state.context is not None
    assert result.state.context.query_results.get("nutrition") is not None
