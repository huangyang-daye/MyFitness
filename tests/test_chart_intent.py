"""图表意图与对话链路测试 — router 分类、chat 分支、日报区间追问。"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, BodyMetric, User
from myfitness.graph.chat import new_chat_state, run_chat_turn
from myfitness.graph.router import classify_intent
from myfitness.schemas.state import Intent


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DAILY_REPORT_OUTPUT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("CHART_OUTPUT_DIR", str(tmp_path / "charts"))
    from myfitness.config import get_settings

    get_settings.cache_clear()

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()

    today = date.today()
    for i in range(10):
        session.add(
            BodyMetric(
                user_id=1,
                record_date=today - timedelta(days=9 - i),
                metric_type="weight",
                value=72.0 - i * 0.2,
                unit="kg",
                source="manual",
                xunji_ref=f"w{i}",
            )
        )
    session.flush()
    yield session
    session.close()


# --- 意图识别 ---


def test_router_classifies_chart_trigger():
    result = classify_intent("生成最近7天体重折线图", use_llm=False)
    assert result.intent == Intent.CHART_TRIGGER
    assert result.domain == "body"
    assert result.start_date == date.today() - timedelta(days=6)
    assert result.end_date == date.today()


@pytest.mark.parametrize(
    "text",
    [
        "近30天体脂趋势图",
        "把摄入热量画成柱状图",
        "给我一个训练容量的统计图",
        "最近7天体重曲线图",
        "可视化一下我的体重变化",
    ],
)
def test_router_chart_keyword_variants(text):
    assert classify_intent(text, use_llm=False).intent == Intent.CHART_TRIGGER


def test_router_trend_analysis_without_chart_words():
    result = classify_intent("近30天体重变化趋势", use_llm=False)
    assert result.intent == Intent.TREND_ANALYSIS


def test_router_still_recognizes_report_and_schedule():
    assert (
        classify_intent("生成昨天的日报", use_llm=False).intent == Intent.REPORT_TRIGGER
    )
    assert (
        classify_intent("每天早上7点生成日报", use_llm=False).intent
        == Intent.SCHEDULE_MANAGE
    )


def test_router_report_period_range():
    today = date(2026, 8, 28)
    result = classify_intent("生成8月20日到8月25日的报告", use_llm=False, today=today)
    assert result.intent == Intent.REPORT_TRIGGER
    assert result.start_date == date(2026, 8, 20)
    assert result.end_date == date(2026, 8, 25)


def test_router_report_plus_chart_combo():
    result = classify_intent("生成最近7天的报告并附上体重折线图", use_llm=False)
    assert Intent.REPORT_TRIGGER in result.intents
    assert Intent.CHART_TRIGGER in result.intents


def test_router_past_n_days():
    today = date(2026, 8, 28)
    result = classify_intent("同步前7天数据", use_llm=False, today=today)
    assert result.start_date == date(2026, 8, 22)
    assert result.end_date == today


# --- 对话链路 ---


def _chat(db_session, message, state=None):
    state = state or new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        return run_chat_turn(db_session, state, message)


def test_chat_inline_chart_reply_contains_mermaid(db_session):
    state = _chat(db_session, "生成前7天的体重折线图")

    assert state.intent == Intent.CHART_TRIGGER
    assert "```mermaid" in state.reply
    assert "xychart-beta" in state.reply
    assert 'y-axis "体重 (kg)"' in state.reply
    assert "chart_tools" in state.metadata.tools_invoked


def test_chat_chart_document_mode_writes_file(db_session):
    state = _chat(db_session, "把最近7天体重折线图保存成文档")

    assert "已生成统计图文档" in state.reply
    assert "```mermaid" in state.reply
    path = state.reply.split("已生成统计图文档：")[1].splitlines()[0].strip()
    from pathlib import Path

    assert Path(path).exists()
    assert "xychart-beta" in Path(path).read_text(encoding="utf-8")


def test_chat_chart_insert_into_existing_report(db_session, tmp_path):
    report = tmp_path / "reports" / f"{date.today().isoformat()}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# MyFitness 日报 — {date.today().isoformat()}\n\n## 分析摘要\n\n正文\n",
        encoding="utf-8",
    )

    state = _chat(db_session, f"把最近7天体重折线图插入到今天的日报")

    assert "已把统计图插入文档" in state.reply
    content = report.read_text(encoding="utf-8")
    assert "xychart-beta" in content
    assert content.index("## 分析摘要") < content.index("xychart-beta")


def test_chat_period_report_with_chart_inserted(db_session, tmp_path):
    state = _chat(db_session, "生成最近3天的报告并附上体重折线图")

    assert "周期报表" in state.reply
    files = list((tmp_path / "reports").glob("*_*.md"))
    assert files, "应生成区间周期报表文件"
    content = files[0].read_text(encoding="utf-8")
    assert "xychart-beta" in content


def test_chat_report_clarification_accepts_range(db_session):
    state = _chat(db_session, "生成日报")
    assert state.pending_confirmation is not None
    assert "哪天" in state.reply

    state = _chat(db_session, "2026-08-20 到 2026-08-25", state=state)
    assert state.pending_confirmation is None
    assert "周期报表" in state.reply
    assert "2026-08-20 ~ 2026-08-25" in state.reply


def test_chat_single_point_chart_explains_instead_of_drawing(db_session):
    state = _chat(db_session, f"生成 {date.today().isoformat()} 的体重折线图")
    assert "1 个数据点" in state.reply
    assert "```mermaid" not in state.reply
