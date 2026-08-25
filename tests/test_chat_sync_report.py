"""对话链路的同步日期与「同步+日报」组合意图测试。"""

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, User
from myfitness.graph.chat import new_chat_state, run_chat_turn


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


def test_chat_sync_today_uses_today_range(db_session):
    """「同步今日数据」只同步今天，而不是最近 7 天。"""
    today = date.today()
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        with patch("myfitness.graph.chat.run_sync") as sync_mock:
            sync_mock.return_value = {
                "status": "success",
                "start_date": today.isoformat(),
                "end_date": today.isoformat(),
                "results": {},
                "errors": [],
            }
            state = run_chat_turn(db_session, state, "同步今日数据")

    kwargs = sync_mock.call_args.kwargs
    assert kwargs.get("start_date") == today
    assert kwargs.get("end_date") == today
    assert kwargs.get("days") is None
    assert "同步完成" in state.reply


def test_chat_sync_without_date_keeps_default_days(db_session):
    """未指明日期的同步保持默认（最近 7 天）。"""
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        with patch("myfitness.graph.chat.run_sync") as sync_mock:
            sync_mock.return_value = {
                "status": "success",
                "start_date": "2026-08-18",
                "end_date": "2026-08-24",
                "results": {},
                "errors": [],
            }
            state = run_chat_turn(db_session, state, "帮我同步训记数据")

    assert sync_mock.call_args.kwargs.get("days") == 7


def test_chat_sync_and_report_combo_syncs_first(db_session):
    """「同步8月24日数据并生成日报」先同步该日数据，再生成该日日报。"""
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_sync") as sync_mock,
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        sync_mock.return_value = {
            "status": "success",
            "start_date": "2026-08-24",
            "end_date": "2026-08-24",
            "results": {},
            "errors": [],
        }
        report_mock.return_value = {
            "report_date": "2026-08-24",
            "file_path": "reports/2026-08-24.md",
            "content_md": "# MyFitness 日报 — 2026-08-24",
        }
        state = run_chat_turn(db_session, state, "同步8月24日数据并生成日报")

    # 先同步：明确日期范围，且不使用 days 默认值
    sync_kwargs = sync_mock.call_args.kwargs
    assert sync_kwargs.get("start_date") == date(2026, 8, 24)
    assert sync_kwargs.get("end_date") == date(2026, 8, 24)
    assert sync_kwargs.get("days") is None

    # 再生成日报：日期与同步一致，且不再重复同步
    report_kwargs = report_mock.call_args.kwargs
    assert report_kwargs.get("report_date") == date(2026, 8, 24)
    assert report_kwargs.get("sync_first") is False

    assert "同步完成" in state.reply
    assert "已生成" in state.reply
    assert "2026-08-24" in state.reply


def test_chat_sync_and_report_combo_report_failure_reports_sync(db_session):
    """组合意图中日报失败时，同步结果仍在回复中体现。"""
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_sync") as sync_mock,
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        sync_mock.return_value = {
            "status": "success",
            "start_date": "2026-08-24",
            "end_date": "2026-08-24",
            "results": {},
            "errors": [],
        }
        report_mock.side_effect = RuntimeError("报告生成崩溃")
        state = run_chat_turn(db_session, state, "同步8月24日数据并生成日报")

    assert "同步完成" in state.reply
    assert "生成日报失败" in state.reply
    assert state.errors


def test_chat_report_uses_route_date(db_session):
    """纯日报意图使用意图识别解析出的日期。"""
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        report_mock.return_value = {
            "report_date": "2026-08-21",
            "file_path": "/tmp/report.md",
            "content_md": "# MyFitness 日报 — 2026-08-21",
        }
        state = run_chat_turn(db_session, state, "生成8.21的报告")

    report_mock.assert_called_once()
    assert report_mock.call_args.kwargs["report_date"] == date(2026, 8, 21)
    assert "2026-08-21" in state.reply


def test_chat_report_without_date_asks_for_date_then_generates(db_session):
    """「生成日报」未指明日期时先追问，用户补日期后再生成。"""
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        report_mock.return_value = {
            "report_date": "2026-08-24",
            "file_path": "/tmp/report.md",
            "content_md": "# MyFitness 日报 — 2026-08-24",
        }

        state = run_chat_turn(db_session, state, "生成日报")

        report_mock.assert_not_called()
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.action_type == "report_date_clarification"
        assert "哪天" in state.reply

        state = run_chat_turn(db_session, state, "2026-08-24")

    report_mock.assert_called_once()
    assert report_mock.call_args.kwargs["report_date"] == date(2026, 8, 24)
    assert state.pending_confirmation is None
    assert "2026-08-24" in state.reply


def test_chat_sync_and_report_without_date_asks_before_running(db_session):
    """「同步数据并生成日报」缺日期时先追问，不提前同步。"""
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_sync") as sync_mock,
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        sync_mock.return_value = {
            "status": "success",
            "start_date": "2026-08-24",
            "end_date": "2026-08-24",
            "results": {},
            "errors": [],
        }
        report_mock.return_value = {
            "report_date": "2026-08-24",
            "file_path": "reports/2026-08-24.md",
            "content_md": "# MyFitness 日报 — 2026-08-24",
        }

        state = run_chat_turn(db_session, state, "同步数据并生成日报")

        sync_mock.assert_not_called()
        report_mock.assert_not_called()
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.action_type == "sync_report_date_clarification"
        assert "先同步该日数据再生成日报" in state.reply

        state = run_chat_turn(db_session, state, "2026-08-24")

    assert sync_mock.call_args.kwargs["start_date"] == date(2026, 8, 24)
    assert sync_mock.call_args.kwargs["end_date"] == date(2026, 8, 24)
    assert report_mock.call_args.kwargs["report_date"] == date(2026, 8, 24)
    assert state.pending_confirmation is None
    assert "同步完成" in state.reply
    assert "已生成" in state.reply


def test_chat_llm_multi_intent_route_to_combo(db_session):
    """LLM 意图 Agent 识别出多意图时，同样走「先同步再日报」链路。"""
    from myfitness.schemas.state import RouteResult

    llm_route = RouteResult(
        intents=["sync_trigger", "report_trigger"],
        start_date=date(2026, 8, 24),
        end_date=date(2026, 8, 24),
    )
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=True),
        patch("myfitness.agents.intent_agent.run_intent_agent", return_value=llm_route),
        patch("myfitness.graph.chat.run_sync") as sync_mock,
        patch("myfitness.graph.chat.run_daily_report") as report_mock,
    ):
        sync_mock.return_value = {
            "status": "success",
            "start_date": "2026-08-24",
            "end_date": "2026-08-24",
            "results": {},
            "errors": [],
        }
        report_mock.return_value = {
            "report_date": "2026-08-24",
            "file_path": "reports/2026-08-24.md",
            "content_md": "# 日报",
        }
        state = run_chat_turn(db_session, state, "同步一下8月24日的数据，顺便把日报出了")

    assert sync_mock.call_args.kwargs.get("start_date") == date(2026, 8, 24)
    assert report_mock.call_args.kwargs.get("report_date") == date(2026, 8, 24)
    assert "同步完成" in state.reply
    assert "已生成" in state.reply
