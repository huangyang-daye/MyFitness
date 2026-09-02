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


def test_parse_single_date_ignores_decimal_ratios():
    today = date(2026, 9, 1)
    assert parse_single_date("0.5倍体重脂肪", today) is None
    assert parse_single_date("2倍体重蛋白质", today) is None
    assert parse_single_date("72.5kg", today) is None


def test_build_query_plan_macro_ratios_not_parsed_as_dates():
    today = date(2026, 9, 1)
    message = (
        "参考我的身体数据，根据3倍体重碳水，2倍体重蛋白质，0.5倍体重脂肪的原则，"
        "规划一下每日的饮食构成，生成规划然后输出为文档"
    )
    plan = build_query_plan(message, Intent.TREND_ANALYSIS, domain="nutrition", today=today)
    assert plan is not None
    assert plan.end_date == today
    assert "nutrition" in plan.domains


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


def test_build_query_plan_progress_until_today_not_single_day():
    today = date(2026, 9, 1)
    plan = build_query_plan(
        "评价一下我减肥到今天的进度怎么样",
        Intent.TREND_ANALYSIS,
        today=today,
    )
    assert plan is not None
    assert plan.start_date == date(2026, 8, 3)
    assert plan.end_date == today
    assert plan.lookback_days == 30


def test_build_query_plan_back_day_uses_history_not_today_only():
    """「今天练背 + 参考过往记录」应查近 30 天训练，而非仅今天。"""
    today = date(2026, 9, 2)
    message = "按照安排，今天是练背日，根据我过往的训练记录，帮我安排一下今天的计划"
    plan = build_query_plan(message, Intent.GENERAL, today=today)
    assert plan is not None
    assert plan.start_date == date(2026, 8, 4)
    assert plan.end_date == today
    assert "training" in plan.domains


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


def test_widen_progress_plan_to_earliest_body_metric(db_session):
    today = date(2026, 9, 1)
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2025, 9, 1),
            metric_type="weight",
            value=130,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=today,
            metric_type="weight",
            value=120,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    from myfitness.services.context_with_query import load_context_for_turn

    context, tools = load_context_for_turn(
        db_session,
        1,
        "评价一下我减肥到今天的进度怎么样",
        Intent.TREND_ANALYSIS,
        domain="body",
        end_date=today,
    )
    assert "query_body_metrics" in tools
    body = context.query_results["body"]
    assert body["count"] == 2
    assert body["start_date"] == "2025-09-01"
    assert body["end_date"] == "2026-09-01"


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
