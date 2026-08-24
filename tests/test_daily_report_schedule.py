"""日报与定时任务测试。"""

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.schedule_parser import parse_schedule_request
from myfitness.db.models import Base, BodyMetric, TrainingLog, User
from myfitness.db.repositories.metrics import SOURCE_MANUAL
from myfitness.db.repositories.reports import DailyReportRepository, ScheduledTaskRepository
from myfitness.graph.chat import new_chat_state, run_chat_turn
from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent
from myfitness.services.daily_report import format_daily_report_md, run_daily_report

SAMPLE_TRAINING_PAYLOAD = {
    "localid": 1787305387305,
    "datestr": "2026-08-22",
    "title": "腿臀",
    "note": "calorie:269 personalworkout_id:1786999087642 personalplanid:286198",
    "movements": [
        {
            "index": 1,
            "name": "单腿哑铃硬拉",
            "type": "臀部",
            "sets": [
                {"index": 1, "done": True, "weight": "12.5", "unit": "kg", "reps": "12", "time": 60},
                {"index": 2, "done": True, "weight": "12.5", "unit": "kg", "reps": "12", "time": 60},
            ],
        },
    ],
}


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    yield session
    session.close()


def test_parse_schedule_create_daily_report():
    parsed = parse_schedule_request("每天早上7点生成日报")
    assert parsed is not None
    assert parsed["action"] == "upsert"
    assert parsed["task_type"] == "daily_report"
    assert parsed["time_of_day"] == "07:00"


def test_parse_schedule_list():
    assert parse_schedule_request("查看定时任务")["action"] == "list"


def test_parse_schedule_cancel():
    parsed = parse_schedule_request("取消日报定时任务")
    assert parsed["action"] == "cancel"
    assert parsed["task_type"] == "daily_report"


def test_classify_schedule_and_report_intents():
    assert classify_intent("每天早上7点生成日报", use_llm=False).intent == Intent.SCHEDULE_MANAGE
    assert classify_intent("生成昨天日报", use_llm=False).intent == Intent.REPORT_TRIGGER
    assert classify_intent("生成8.21的报告", use_llm=False).intent == Intent.REPORT_TRIGGER
    assert classify_intent("查看定时任务", use_llm=False).intent == Intent.SCHEDULE_MANAGE


def test_run_daily_report_persists(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("DAILY_REPORT_OUTPUT_DIR", str(tmp_path))
    from myfitness.config import get_settings

    get_settings.cache_clear()
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 22),
            metric_type="weight",
            value=72.0,
            unit="kg",
            source=SOURCE_MANUAL,
        )
    )
    db_session.flush()

    with patch("myfitness.services.daily_report.run_sync") as sync_mock:
        sync_mock.return_value = {"status": "success"}
        result = run_daily_report(
            db_session,
            1,
            report_date=date(2026, 8, 22),
            sync_first=True,
        )

    assert result["report_date"] == "2026-08-22"
    saved = DailyReportRepository(db_session, 1).get_by_date(date(2026, 8, 22))
    assert saved is not None
    assert "MyFitness 日报" in saved.content_md
    assert "报告日训练次数" not in saved.content_md


def test_daily_report_training_section_shows_detail(db_session):
    db_session.add(
        TrainingLog(
            user_id=1,
            record_date=date(2026, 8, 22),
            title="腿臀",
            raw_payload=SAMPLE_TRAINING_PAYLOAD,
            source="xunji_sync",
            xunji_localid="1787305387305",
        )
    )
    db_session.flush()

    with patch("myfitness.services.daily_report.run_sync") as sync_mock:
        sync_mock.return_value = {"status": "success"}
        result = run_daily_report(
            db_session,
            1,
            report_date=date(2026, 8, 22),
            sync_first=False,
        )

    content = result["content_md"]
    assert "### 训练（报告日）" in content
    assert "报告日训练次数" not in content
    assert "腿臀" in content
    assert "单腿哑铃硬拉" in content
    assert "12.5kg×12" in content


def test_format_daily_report_md_no_training():
    from myfitness.schemas.state import ContextSnapshot, DateRange

    context = ContextSnapshot(
        date_range=DateRange(start=date(2026, 8, 22), end=date(2026, 8, 22)),
        body_metrics_summary={},
        nutrition_summary={"today_totals": {}},
        training_summary={},
    )
    md = format_daily_report_md(
        report_date=date(2026, 8, 22),
        context=context,
        summary_content="测试摘要",
        training_sessions=[],
    )
    assert "报告日无训练记录。" in md
    assert "报告日训练次数" not in md


def test_chat_schedule_confirm_flow(db_session):
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        state = run_chat_turn(db_session, state, "每天早上7点生成日报")
        assert state.pending_confirmation is not None
        assert state.pending_confirmation.action_type == "schedule_upsert"
        state = run_chat_turn(db_session, state, "确认")

    task = ScheduledTaskRepository(db_session, 1).get_by_type("daily_report")
    assert task is not None
    assert task.time_of_day == "07:00"
    assert task.enabled is True


def test_chat_report_uses_requested_date(db_session):
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        with patch("myfitness.graph.chat.run_daily_report") as report_mock:
            report_mock.return_value = {
                "report_date": "2026-08-21",
                "file_path": "/tmp/report.md",
                "content_md": "# MyFitness 日报 — 2026-08-21",
            }
            state = run_chat_turn(db_session, state, "生成8.21的报告")

    report_mock.assert_called_once()
    assert report_mock.call_args.kwargs["report_date"] == date(2026, 8, 21)
    assert "2026-08-21" in state.reply
