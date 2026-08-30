"""周期报表测试 — 单日退化为日报，多日追加趋势图与每日明细。"""

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, BodyMetric, NutritionLog, TrainingLog, User
from myfitness.db.repositories.reports import DailyReportRepository
from myfitness.services.period_report import (
    format_daily_breakdown,
    format_period_highlights,
    normalize_period,
    run_period_report,
)

START = date(2026, 8, 20)
END = date(2026, 8, 26)


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DAILY_REPORT_OUTPUT_DIR", str(tmp_path))
    from myfitness.config import get_settings

    get_settings.cache_clear()

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()

    for i in range(7):
        d = START + timedelta(days=i)
        session.add(
            BodyMetric(
                user_id=1,
                record_date=d,
                metric_type="weight",
                value=72.4 - i * 0.2,
                unit="kg",
                source="manual",
                xunji_ref=f"w{i}",
            )
        )
        session.add(
            BodyMetric(
                user_id=1,
                record_date=d,
                metric_type="bodyfat",
                value=18.4 - i * 0.1,
                unit="%",
                source="manual",
                xunji_ref=f"f{i}",
            )
        )
        session.add(
            NutritionLog(
                user_id=1,
                record_date=d,
                meal_type="lunch",
                food_name="鸡胸肉",
                amount=200,
                unit="g",
                nutrients_snapshot={"cal": 330, "protein": 62, "fat": 7, "carb": 0},
                source="manual",
            )
        )
    for i in (0, 3, 6):
        session.add(
            TrainingLog(
                user_id=1,
                record_date=START + timedelta(days=i),
                title="胸",
                raw_payload={
                    "title": "胸",
                    "movements": [
                        {"name": "卧推", "sets": [{"weight": "60", "unit": "kg", "reps": "10"}]}
                    ],
                },
                source="xunji_sync",
                xunji_localid=f"t{i}",
            )
        )
    session.flush()
    yield session
    session.close()


def test_normalize_period_fills_missing_side():
    assert normalize_period(None, date(2026, 8, 20)) == (date(2026, 8, 20), date(2026, 8, 20))
    assert normalize_period(date(2026, 8, 25), date(2026, 8, 20)) == (
        date(2026, 8, 20),
        date(2026, 8, 25),
    )
    assert normalize_period(None, None) == (date.today() - timedelta(days=1),) * 2


def test_single_day_report_degrades_to_daily_report(db_session):
    result = run_period_report(db_session, 1, START, START, sync_first=False)

    assert result["report_kind"] == "daily"
    assert result["period_days"] == 1
    assert result["report_date"] == START.isoformat()
    content = result["content_md"]
    assert "# MyFitness 日报 — 2026-08-20" in content
    assert "xychart-beta" not in content
    assert result["file_path"].endswith("2026-08-20.md")


def test_multi_day_report_is_period_report_with_charts(db_session):
    result = run_period_report(db_session, 1, START, END, sync_first=False)

    assert result["report_kind"] == "period"
    assert result["period_days"] == 7
    assert result["period_start"] == START.isoformat()
    assert result["period_end"] == END.isoformat()
    assert [c["metric"] for c in result["charts"]][:2] == ["weight", "bodyfat"]

    content = result["content_md"]
    assert "# MyFitness 周期报表 — 2026-08-20 ~ 2026-08-26（7 天）" in content
    assert "## 身体数据趋势" in content
    assert "xychart-beta" in content
    assert 'x-axis ["08-20", "08-21"' in content
    assert "line [72.4, 72.2, 72, 71.8, 71.6, 71.4, 71.2]" in content
    assert "## 每日明细" in content
    assert "| 日期 | 体重 (kg) | 体脂率 (%) | 热量 (kcal) | 蛋白质 (g) | 训练次数 |" in content
    assert "## 区间汇总" in content
    assert "体重：72.4 → 71.2 kg（-1.2 kg，7 个记录日）" in content
    assert "训练：3 次" in content
    assert result["file_path"].endswith("2026-08-20_2026-08-26.md")


def test_period_report_persists_period_metadata(db_session):
    result = run_period_report(db_session, 1, START, END, sync_first=False)
    saved = DailyReportRepository(db_session, 1).get_by_date(END)

    assert saved is not None
    payload = saved.agent_outputs or {}
    assert payload["period"]["start_date"] == START.isoformat()
    assert payload["period"]["end_date"] == END.isoformat()
    assert payload["period"]["report_kind"] == "period"
    assert payload["period"]["charts"] == [c["metric"] for c in result["charts"]]


def test_period_report_without_enough_body_data_skips_chart(db_session):
    empty_start = date(2026, 7, 1)
    empty_end = date(2026, 7, 5)
    result = run_period_report(db_session, 1, empty_start, empty_end, sync_first=False)

    assert result["charts"] == []
    assert "区间内身体指标数据点不足" in result["content_md"]
    assert "xychart-beta" not in result["content_md"]


def test_period_report_sync_failure_does_not_break(db_session, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sync down")

    monkeypatch.setattr("myfitness.services.period_report.run_sync", boom)
    result = run_period_report(db_session, 1, START, END, sync_first=True)

    assert result["sync"]["status"] == "failed"
    assert "MyFitness 周期报表" in result["content_md"]


def test_format_daily_breakdown_marks_missing_days(db_session):
    from myfitness.agents.tools.base import invoke_tool
    from myfitness.agents.tools.query_tools import (
        query_body_metrics,
        query_nutrition_logs,
        query_training_logs,
    )

    body = invoke_tool(
        query_body_metrics, db_session, 1, start_date=START, end_date=START + timedelta(days=2)
    ).get("records") or []
    nutrition = invoke_tool(
        query_nutrition_logs, db_session, 1, start_date=START, end_date=START + timedelta(days=2)
    )
    training = invoke_tool(
        query_training_logs, db_session, 1, start_date=START, end_date=START + timedelta(days=2)
    ).get("sessions") or []

    table = format_daily_breakdown(
        START, START + timedelta(days=2), body, nutrition, training
    )
    lines = table.splitlines()
    assert "| 2026-08-20 | 72.4 | 18.4 | 330 | 62 | 1 |" in lines
    # 8/22 无训练 → 0
    assert "| 2026-08-22 | 72 | 18.2 | 330 | 62 | 0 |" in lines


def test_format_period_highlights_without_data():
    text = format_period_highlights([], {}, [], 7)
    assert "饮食：区间内无饮食记录" in text
    assert "训练：区间内无训练记录" in text
    assert "体重" not in text
