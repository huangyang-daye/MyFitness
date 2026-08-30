"""统计图 Tool 测试 — 数据聚合、Mermaid 渲染、文档生成与插入。"""

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.chart_tools import (
    ChartRequest,
    aggregate_body_series,
    aggregate_nutrition_series,
    aggregate_training_series,
    build_body_metric_chart,
    build_body_trend_charts,
    build_chart,
    build_nutrition_chart,
    build_training_chart,
    default_chart_filename,
    insert_chart_into_document,
    parse_chart_request,
    resolve_metric,
    write_chart_document,
)
from myfitness.db.models import Base, BodyMetric, NutritionLog, TrainingLog, User

START = date(2026, 8, 20)
END = date(2026, 8, 26)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()

    weights = [72.4, 72.1, 72.0, 71.6, 71.8, 71.2, 70.9]
    for i, w in enumerate(weights):
        d = START + timedelta(days=i)
        session.add(
            BodyMetric(
                user_id=1,
                record_date=d,
                metric_type="weight",
                value=w,
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
    # 8/22 无饮食记录，验证缺失日补 0
    for i in (0, 1, 3, 4, 5, 6):
        session.add(
            NutritionLog(
                user_id=1,
                record_date=START + timedelta(days=i),
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
                        {
                            "name": "卧推",
                            "sets": [
                                {"weight": "60", "unit": "kg", "reps": "10"},
                                {"weight": "60", "unit": "kg", "reps": "8"},
                            ],
                        }
                    ],
                },
                source="xunji_sync",
                xunji_localid=f"t{i}",
            )
        )
    session.flush()
    yield session
    session.close()


# --- 数据聚合 ---


def test_aggregate_body_series_skips_missing_days(db_session):
    labels, values = aggregate_body_series(db_session, 1, START, END, "weight")
    assert len(labels) == 7
    assert labels[0] == "08-20"
    assert values[0] == 72.4
    assert values[-1] == 70.9


def test_aggregate_body_series_averages_same_day_sources(db_session):
    db_session.add(
        BodyMetric(
            user_id=1,
            record_date=START,
            metric_type="weight",
            value=70.0,
            unit="kg",
            source="xunji_sync",
            xunji_ref="dup",
        )
    )
    db_session.flush()
    _, values = aggregate_body_series(db_session, 1, START, START, "weight")
    assert values == [pytest.approx(71.2)]  # (72.4 + 70.0) / 2


def test_aggregate_nutrition_series_fills_missing_with_zero(db_session):
    labels, values = aggregate_nutrition_series(db_session, 1, START, END, "calories")
    assert len(labels) == 7
    assert values[2] == 0  # 8/22 无记录
    assert values[0] == 330


def test_aggregate_training_series_counts_sessions(db_session):
    labels, values = aggregate_training_series(db_session, 1, START, END, "sessions")
    assert labels == ["08-20", "08-23", "08-26"]
    assert values == [1, 1, 1]

    _, volume = aggregate_training_series(db_session, 1, START, END, "volume_kg")
    assert volume == [1080.0, 1080.0, 1080.0]  # 60*10 + 60*8


# --- Mermaid 渲染 ---


def test_chart_mermaid_has_date_axis_and_values(db_session):
    chart = invoke_tool(build_body_metric_chart, db_session, 1, start_date=START, end_date=END, metric_type="weight")
    mermaid = chart.to_mermaid()

    assert mermaid.startswith("```mermaid")
    assert "xychart-beta" in mermaid
    assert 'x-axis ["08-20", "08-21"' in mermaid
    assert 'y-axis "体重 (kg)"' in mermaid
    assert "line [72.4, 72.1, 72, 71.6, 71.8, 71.2, 70.9]" in mermaid
    assert "-->" in mermaid  # y 轴范围


def test_chart_mermaid_bar_type_for_nutrition(db_session):
    chart = invoke_tool(build_nutrition_chart, db_session, 1, start_date=START, end_date=END, field_name="calories")
    mermaid = chart.to_mermaid()
    assert "bar [" in mermaid
    assert 'y-axis "热量 (kcal)"' in mermaid


def test_chart_training_uses_volume(db_session):
    chart = invoke_tool(build_training_chart, db_session, 1, start_date=START, end_date=END, field_name="volume_kg")
    assert "总容量" in chart.title
    assert chart.to_mermaid() is not None


def test_chart_y_axis_lower_bound_never_negative(db_session):
    """热量 / 容量类指标含 0 时，y 轴下界不应出现负数。"""
    chart = invoke_tool(build_nutrition_chart, db_session, 1, start_date=START, end_date=END, field_name="calories")
    mermaid = chart.to_mermaid()
    y_line = next(line for line in mermaid.splitlines() if "y-axis" in line)
    lower = float(y_line.split("-->")[0].split('"')[-1].strip())
    assert lower >= 0


def test_chart_empty_when_no_data(db_session):
    chart = invoke_tool(build_body_metric_chart, db_session, 1, start_date=date(2020, 1, 1), end_date=date(2020, 1, 10), metric_type="weight")
    assert chart.is_empty
    assert chart.to_mermaid() is None
    assert "暂无足够数据" in chart.to_markdown()


def test_chart_downsample_keeps_first_and_last(db_session):
    chart = invoke_tool(build_body_metric_chart, db_session, 1, start_date=START, end_date=END, metric_type="weight", max_points=3)
    assert len(chart.x_labels) <= 4
    assert chart.x_labels[0] == "08-20"
    assert chart.x_labels[-1] == "08-26"


def test_chart_table_and_summary(db_session):
    chart = invoke_tool(build_body_metric_chart, db_session, 1, start_date=START, end_date=END, metric_type="weight")
    table = chart.to_table()
    assert "| 日期 | 体重 (kg) |" in table
    assert "| 08-20 | 72.4 |" in table
    assert "70.9" in chart.summary_line()
    assert "-1.5" in chart.summary_line()


def test_build_body_trend_charts_only_keeps_multi_point_metrics(db_session):
    # 只给 weight 一个单点数据，bodyfat 有多天
    db_session.query(BodyMetric).filter(
        BodyMetric.record_date != START, BodyMetric.metric_type == "weight"
    ).delete()
    db_session.flush()

    charts = invoke_tool(build_body_trend_charts, db_session, 1, start_date=START, end_date=END)
    metrics = [c.metric for c in charts]
    assert "bodyfat" in metrics
    assert "weight" not in metrics


def test_build_chart_dispatches_by_domain(db_session):
    request = ChartRequest(
        domain="training", metric="sessions", start_date=START, end_date=END, chart_type="bar"
    )
    chart = build_chart(db_session, 1, request)
    assert chart.domain == "training"
    assert "训练次数" in chart.title


# --- 请求解析 ---


def test_parse_chart_request_weight_recent_days():
    today = date(2026, 8, 28)
    req = parse_chart_request("生成最近7天体重折线图", today=today)
    assert req.domain == "body"
    assert req.metric == "weight"
    assert req.chart_type == "line"
    assert req.start_date == date(2026, 8, 22)
    assert req.end_date == today
    assert req.output_mode == "inline"


def test_parse_chart_request_past_n_days_and_bar():
    today = date(2026, 8, 28)
    req = parse_chart_request("前30天摄入热量柱状图", today=today)
    assert req.domain == "nutrition"
    assert req.metric == "calories"
    assert req.chart_type == "bar"
    assert req.start_date == date(2026, 7, 30)


def test_parse_chart_request_explicit_range():
    req = parse_chart_request("生成8月20日到8月25日体脂趋势图", today=date(2026, 8, 28))
    assert req.metric == "bodyfat"
    assert req.start_date == date(2026, 8, 20)
    assert req.end_date == date(2026, 8, 25)


def test_parse_chart_request_document_mode():
    req = parse_chart_request("把最近7天体重折线图保存成文档", today=date(2026, 8, 28))
    assert req.output_mode == "document"


def test_parse_chart_request_insert_mode_resolves_report(tmp_path):
    report = tmp_path / "2026-08-24.md"
    report.write_text("# MyFitness 日报 — 2026-08-24\n", encoding="utf-8")
    req = parse_chart_request("把体重折线图插入到8月24日的日报", today=date(2026, 8, 28), reports_dir=tmp_path)
    assert req.output_mode == "insert"
    assert req.target_path == report


def test_parse_chart_request_training_volume():
    req = parse_chart_request("最近14天训练容量柱状图", today=date(2026, 8, 28))
    assert req.domain == "training"
    assert req.metric == "volume_kg"


def test_resolve_metric_unknown_falls_back_to_weight():
    assert resolve_metric("weight") == ("body", "体重", "kg")
    assert resolve_metric("calories") == ("nutrition", "热量", "kcal")
    assert resolve_metric("sessions") == ("training", "训练次数", "次")
    assert resolve_metric("unknown_metric")[0] == "body"


# --- 文档输出 ---


def test_write_chart_document_creates_markdown(db_session, tmp_path):
    chart = invoke_tool(build_body_metric_chart, db_session, 1, start_date=START, end_date=END, metric_type="weight")
    path = write_chart_document(chart, tmp_path / "charts")

    assert path.exists()
    assert path.name == default_chart_filename(chart)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# 体重趋势")
    assert "```mermaid" in content
    assert "数据范围：2026-08-20 ~ 2026-08-26" in content


def test_insert_chart_appends_when_no_anchor(db_session, tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text("# 日报\n\n## 分析摘要\n\n正文\n", encoding="utf-8")

    invoke_tool(insert_chart_into_document, db_session, 1, path=str(doc), markdown="### 体重趋势\n\n```mermaid\nxychart-beta\n```")

    content = doc.read_text(encoding="utf-8")
    assert content.index("## 分析摘要") < content.index("### 体重趋势")
    assert "正文" in content


def test_insert_chart_into_section_by_anchor(db_session, tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text(
        "# 日报\n\n## 身体数据趋势\n\n旧内容\n\n## 每日明细\n\n| 日期 |\n",
        encoding="utf-8",
    )

    invoke_tool(insert_chart_into_document, db_session, 1, path=str(doc), markdown="### 新图表", anchor="## 身体数据趋势")

    content = doc.read_text(encoding="utf-8")
    assert content.index("### 新图表") < content.index("## 每日明细")
    assert content.index("## 身体数据趋势") < content.index("### 新图表")


def test_insert_chart_creates_missing_file(db_session, tmp_path):
    doc = tmp_path / "new.md"
    invoke_tool(insert_chart_into_document, db_session, 1, path=str(doc), markdown="### 图", create_if_missing=True)
    assert doc.exists()
    assert "### 图" in doc.read_text(encoding="utf-8")
