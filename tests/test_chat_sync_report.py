"""对话链路的同步日期与「同步+日报」组合意图测试。"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, User
from myfitness.graph.chat import new_chat_state, run_chat_turn
from myfitness.schemas.state import ChatMessage, Intent


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


def test_chat_sync_failed_status_is_not_reported_as_complete(db_session):
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_sync") as sync_mock,
    ):
        sync_mock.return_value = {
            "status": "failed",
            "start_date": "2026-08-24",
            "end_date": "2026-08-30",
            "results": {},
            "errors": ["body: 当前进程没有外网套接字访问权限（WinError 10013）。"],
        }
        state = run_chat_turn(db_session, state, "帮我同步训记数据")

    assert "同步失败" in state.reply
    assert "WinError 10013" in state.reply
    assert "同步完成" not in state.reply


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


def test_chat_report_without_context_asks_for_date_then_generates(db_session):
    """没有可继承的分析上下文时先追问，用户补日期后再生成。"""
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

        state = run_chat_turn(db_session, state, "生成报告")

        report_mock.assert_not_called()
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.action_type == "report_date_clarification"
        assert "哪天" in state.reply

        state = run_chat_turn(db_session, state, "2026-08-24")

    report_mock.assert_called_once()
    assert report_mock.call_args.kwargs["report_date"] == date(2026, 8, 24)
    assert state.pending_confirmation is None
    assert "2026-08-24" in state.reply


@pytest.mark.parametrize(
    "followup",
    ["生成报告", "根据对话记录生成报告", "整理成报告"],
)
def test_chat_contextual_report_exports_previous_analysis(db_session, followup):
    """分析/建议后的模糊报告指令应导出上一轮内容，而不是追问日报日期。"""
    previous_reply = "根据最近训练记录，建议下周降低深蹲总量并增加一天恢复。"
    state = new_chat_state(user_id=1)
    state.intent = Intent.TREND_ANALYSIS
    state.intent_domain = "fitness"
    state.messages = [
        ChatMessage(role="user", content="根据最近的训练记录，给我一些训练建议"),
        ChatMessage(role="assistant", content=previous_reply),
    ]

    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.apply_document_export") as export_mock,
        patch("myfitness.graph.chat.run_daily_report") as daily_mock,
        patch("myfitness.graph.chat.run_period_report") as period_mock,
    ):
        export_mock.return_value = {
            "document_only": True,
            "exports": [
                {
                    "path": "/tmp/训练分析报告.md",
                    "filename": "训练分析报告.md",
                    "format": "md",
                }
            ],
        }
        state = run_chat_turn(db_session, state, followup)

    export_mock.assert_called_once()
    assert export_mock.call_args.kwargs["source_content"] == previous_reply
    assert "上一轮用户诉求：根据最近的训练记录" in export_mock.call_args.args[2]
    daily_mock.assert_not_called()
    period_mock.assert_not_called()
    assert state.pending_confirmation is None
    assert state.messages[-1].artifacts[0].kind == "document"
    assert "训练分析报告" in state.reply


def test_chat_contextual_report_does_not_inherit_unrelated_turn(db_session):
    """普通闲聊后的“生成报告”仍应追问日期，不能导出无关内容。"""
    state = new_chat_state(user_id=1)
    state.intent = Intent.GENERAL
    state.messages = [
        ChatMessage(role="user", content="你好"),
        ChatMessage(role="assistant", content="你好，有什么可以帮你？"),
    ]

    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.apply_document_export") as export_mock,
    ):
        state = run_chat_turn(db_session, state, "生成报告")

    export_mock.assert_not_called()
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.action_type == "report_date_clarification"


def test_chat_explicit_period_report_ignores_previous_analysis(db_session):
    """显式日期范围始终走周期报表，不应被跨轮导出分支截获。"""
    today = date.today()
    start = today - timedelta(days=6)
    state = new_chat_state(user_id=1)
    state.intent = Intent.TREND_ANALYSIS
    state.intent_domain = "fitness"
    state.messages = [
        ChatMessage(role="user", content="分析最近的训练并给建议"),
        ChatMessage(role="assistant", content="建议安排恢复周。"),
    ]

    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.apply_document_export") as export_mock,
        patch("myfitness.graph.chat.run_period_report") as period_mock,
    ):
        period_mock.return_value = {
            "report_kind": "period",
            "period_start": start.isoformat(),
            "period_end": today.isoformat(),
            "period_days": 7,
            "file_path": "/tmp/period.md",
            "content_md": "# 周期报告",
        }
        state = run_chat_turn(db_session, state, "生成最近一周报告")

    export_mock.assert_not_called()
    period_mock.assert_called_once()
    assert period_mock.call_args.kwargs["start_date"] == start
    assert period_mock.call_args.kwargs["end_date"] == today
    assert state.pending_confirmation is None


def test_chat_report_clarification_accepts_recent_week(db_session):
    """追问日期后回复「最近一周」应生成近 7 天周期报告，而不是继续追问。"""
    state = new_chat_state(user_id=1)
    today = date.today()
    expected_start = today - timedelta(days=6)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.graph.chat.run_period_report") as period_mock,
    ):
        period_mock.return_value = {
            "report_date": expected_start.isoformat(),
            "start_date": expected_start.isoformat(),
            "end_date": today.isoformat(),
            "file_path": "/tmp/period.md",
            "content_md": "# 周期报告",
        }

        state = run_chat_turn(db_session, state, "生成报告")
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.action_type == "report_date_clarification"

        state = run_chat_turn(db_session, state, "最近一周")

    assert state.pending_confirmation is None
    period_mock.assert_called_once()
    assert period_mock.call_args.kwargs["start_date"] == expected_start
    assert period_mock.call_args.kwargs["end_date"] == today
    assert "周期报告" in state.reply or expected_start.isoformat() in state.reply or "已生成" in state.reply


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
