"""对话产物登记：报告 / 统计图文档要挂到助手消息上，供前端渲染卡片。"""

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


def _run_turn(db_session, message, *, report=None, chart=None):
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        with patch("myfitness.graph.chat._generate_report", return_value=report) as report_mock, \
             patch("myfitness.graph.chat.generate_chart", return_value=chart) as chart_mock:
            state = run_chat_turn(db_session, state, message)
    return state, report_mock, chart_mock


def test_daily_report_is_recorded_as_artifact(db_session):
    report = {
        "report_kind": "daily",
        "report_date": "2026-08-29",
        "file_path": "D:/data/reports/2026-08-29.md",
        "content_md": "# 日报",
    }
    state, _, _ = _run_turn(db_session, "生成 2026-08-29 的日报", report=report)

    artifacts = state.messages[-1].artifacts
    assert len(artifacts) == 1
    assert artifacts[0].kind == "report"
    assert artifacts[0].title == "2026-08-29 健康日报"
    assert artifacts[0].path == "D:/data/reports/2026-08-29.md"
    assert state.pending_artifacts == [], "挂到消息后缓冲区必须清空，避免跨轮重复"


def test_period_report_title_covers_range(db_session):
    report = {
        "report_kind": "period",
        "period_start": "2026-08-22",
        "period_end": "2026-08-28",
        "period_days": 7,
        "report_date": "2026-08-28",
        "file_path": "D:/data/reports/2026-08-22_2026-08-28.md",
        "content_md": "# 周期报表",
    }
    state, _, _ = _run_turn(db_session, "生成 2026-08-22 到 2026-08-28 的周期报表", report=report)

    artifact = state.messages[-1].artifacts[0]
    assert artifact.title == "2026-08-22 ~ 2026-08-28 周期报表"
    assert artifact.subtitle == "7 天"


def test_standalone_chart_document_is_recorded(db_session):
    chart = {
        "is_empty": False,
        "metric_label": "体重",
        "output_mode": "document",
        "path": "D:/data/reports/charts/chart-weight.md",
        "markdown": "```mermaid\nline\n```",
        "message": "已生成统计图文档",
        "point_count": 5,
    }
    state, _, chart_mock = _run_turn(db_session, "生成体重的统计图文档", chart=chart)
    assert chart_mock.called

    artifacts = state.messages[-1].artifacts
    assert len(artifacts) == 1
    assert artifacts[0].kind == "chart"
    assert artifacts[0].title == "体重趋势图"


def test_chart_inserted_into_report_is_not_duplicated(db_session):
    """插图模式改的是报告文档，报告卡片已经覆盖，不应再产出一个图表卡片。"""
    report = {
        "report_kind": "daily",
        "report_date": "2026-08-29",
        "file_path": "D:/data/reports/2026-08-29.md",
        "content_md": "# 日报",
    }
    chart = {
        "is_empty": False,
        "metric_label": "体重",
        "output_mode": "insert",
        "path": "D:/data/reports/2026-08-29.md",
        "markdown": "```mermaid\nline\n```",
        "message": "已插入",
        "point_count": 5,
    }
    state, _, _ = _run_turn(
        db_session, "生成 2026-08-29 的日报并插入体重趋势图", report=report, chart=chart
    )

    kinds = [item.kind for item in state.messages[-1].artifacts]
    assert kinds == ["report"]


def test_turn_without_artifacts_has_empty_list(db_session):
    state, _, _ = _run_turn(db_session, "你好", report=None, chart=None)
    assert state.messages[-1].artifacts == []


def test_general_turn_does_not_leak_artifacts(db_session):
    """上一轮的产物不能污染下一轮。"""
    report = {
        "report_kind": "daily",
        "report_date": "2026-08-29",
        "file_path": "D:/data/reports/2026-08-29.md",
        "content_md": "# 日报",
    }
    state, _, _ = _run_turn(db_session, "生成 2026-08-29 的日报", report=report)
    assert state.messages[-1].artifacts

    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        with patch("myfitness.graph.chat._generate_report", return_value=None):
            state = run_chat_turn(db_session, state, "你好")
    assert state.messages[-1].artifacts == []
