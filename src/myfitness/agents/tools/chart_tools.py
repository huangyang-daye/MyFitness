"""统计图 Tool — 用 Mermaid（xychart-beta）渲染折线图 / 柱状图。

职责：
1. 从 DB 聚合出「日期 → 数值」的时间序列（身体指标 / 饮食 / 训练）；
2. 渲染为 Mermaid `xychart-beta` 代码块（横轴日期、纵轴数值）；
3. 按需求输出：对话内联、生成独立 Markdown 文档、或插入已有文档（报表）指定位置。

Mermaid 语法示例（本模块产出）：

```mermaid
xychart-beta
    title "体重趋势（2026-08-20 ~ 2026-08-27）"
    x-axis ["08-20", "08-21"]
    y-axis "体重 (kg)" 71.5 --> 73.2
    line [72.1, 72.4]
```
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.query_tools import (
    query_body_metrics,
    query_nutrition_logs,
    query_training_logs,
)
from myfitness.config import get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 指标字典
# --------------------------------------------------------------------------- #

BODY_METRIC_LABELS = {
    "weight": "体重",
    "bodyfat": "体脂率",
    "neck": "颈围",
    "chest": "胸围",
    "weist": "腰围",
    "shoulder": "肩围",
    "bot": "臀围",
    "arm_left": "左上臂围",
    "arm_right": "右上臂围",
    "forearm_left": "左前臂围",
    "forearm_right": "右前臂围",
    "leg_left": "左腿围",
    "leg_right": "右腿围",
    "cav_left": "左小腿围",
    "cav_right": "右小腿围",
}

BODY_METRIC_UNITS = {
    "weight": "kg",
    "bodyfat": "%",
    "neck": "cm",
    "chest": "cm",
    "weist": "cm",
    "shoulder": "cm",
    "bot": "cm",
    "arm_left": "cm",
    "arm_right": "cm",
    "forearm_left": "cm",
    "forearm_right": "cm",
    "leg_left": "cm",
    "leg_right": "cm",
    "cav_left": "cm",
    "cav_right": "cm",
}

# 周期报表中优先绘制的身体指标（按顺序，其余指标在有数据时补上）
PRIORITY_BODY_METRICS = ("weight", "bodyfat")

NUTRITION_FIELD_LABELS = {
    "calories": ("热量", "kcal"),
    "protein_g": ("蛋白质", "g"),
    "carbs_g": ("碳水", "g"),
    "fat_g": ("脂肪", "g"),
}

TRAINING_FIELD_LABELS = {
    "sessions": ("训练次数", "次"),
    "volume_kg": ("总容量", "kg"),
    "sets": ("总组数", "组"),
    "duration_min": ("训练时长", "min"),
    "calories": ("消耗热量", "kcal"),
}

CHART_TYPE_KEYWORDS = {
    "折线": "line",
    "曲线": "line",
    "趋势": "line",
    "走势": "line",
    "变化": "line",
    "柱状": "bar",
    "柱形": "bar",
    "条形": "bar",
    "bar": "bar",
}

CHART_REQUEST_KEYWORDS = (
    "折线图",
    "曲线图",
    "趋势图",
    "走势图",
    "统计图",
    "图表",
    "画图",
    "画个图",
    "柱状图",
    "柱形图",
    "条形图",
    "可视化",
)

# 默认最多绘制的数据点，超出后等距抽样（保留首末）
DEFAULT_MAX_POINTS = 40
# 折线图至少需要的数据点
MIN_LINE_POINTS = 2

_RECENT_DAYS_RE = re.compile(r"(?:最?\s*近|过[去了]?|前)\s*(\d+)\s*天")
_RANGE_SPLIT_RE = re.compile(r"\s*(?:到|至|~|-|—|～)\s*")
INSERT_DOC_KEYWORDS = ("插入", "加到", "加进", "追加", "放进", "塞进", "补充到")
_DOCUMENT_KEYWORDS = ("文档", "文件", "保存", "导出", "存成", "存为", "写到", "输出成")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class ChartSeries:
    """一条折线/柱 series。"""

    name: str
    values: list[float]
    unit: str = ""

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ChartSpec:
    """一张统计图的完整描述，可渲染为 Mermaid / Markdown。"""

    title: str
    x_labels: list[str]
    series: list[ChartSeries]
    y_label: str = "数值"
    chart_type: str = "line"  # line | bar
    domain: str = "body"
    metric: str = "weight"
    start_date: date | None = None
    end_date: date | None = None
    notes: list[str] = field(default_factory=list)

    # ---- 基本属性 ----
    @property
    def is_empty(self) -> bool:
        return not self.series or all(len(s.values) == 0 for s in self.series)

    @property
    def point_count(self) -> int:
        return max((len(s.values) for s in self.series), default=0)

    @property
    def date_range_label(self) -> str:
        if self.start_date and self.end_date:
            if self.start_date == self.end_date:
                return self.start_date.isoformat()
            return f"{self.start_date.isoformat()} ~ {self.end_date.isoformat()}"
        return ""

    # ---- 渲染 ----
    def value_bounds(self) -> tuple[float, float] | None:
        values = [v for s in self.series for v in s.values if v is not None]
        if not values:
            return None
        return min(values), max(values)

    def to_mermaid(self) -> str | None:
        """渲染为 Mermaid xychart-beta 代码块；数据不足时返回 None。"""
        if self.is_empty:
            return None

        bounds = self.value_bounds()
        if bounds is None:
            return None
        low, high = _pad_bounds(bounds)

        kind = "bar" if self.chart_type == "bar" else "line"
        lines = [
            "```mermaid",
            "xychart-beta",
            f'    title "{_escape_mermaid(self.title)}"',
            "    x-axis [" + ", ".join(f'"{_escape_mermaid(x)}"' for x in self.x_labels) + "]",
            f'    y-axis "{_escape_mermaid(self.y_label)}" {_fmt(low)} --> {_fmt(high)}',
        ]
        for s in self.series:
            if not s.values:
                continue
            values = ", ".join(_fmt(v) for v in s.values)
            lines.append(f"    {kind} [{values}]")
        if len(lines) <= 5:  # 没有任何 series 被渲染
            return None
        lines.append("```")
        return "\n".join(lines)

    def to_table(self) -> str:
        """渲染为 Markdown 数据表（便于无 Mermaid 渲染器时阅读）。"""
        if self.is_empty:
            return "无数据。"
        header = "| 日期 | " + " | ".join(
            f"{s.name}{f' ({s.unit})' if s.unit else ''}" for s in self.series
        )
        lines = [f"{header} |", "|" + "---|" * (len(self.series) + 1)]
        for i, label in enumerate(self.x_labels):
            row = [label]
            for s in self.series:
                row.append(_fmt(s.values[i]) if i < len(s.values) else "—")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_markdown(self, *, include_table: bool = True, heading_level: int = 3) -> str:
        """渲染为可嵌入报表的 Markdown 片段。"""
        mermaid = self.to_mermaid()
        if mermaid is None:
            return f"{'#' * heading_level} {self.title}\n\n暂无足够数据绘制图表。"

        parts = [f"{'#' * heading_level} {self.title}", "", mermaid]
        if self.notes:
            parts += ["", "\n".join(f"- {n}" for n in self.notes)]
        if include_table:
            parts += ["", self.to_table()]
        return "\n".join(parts)

    def summary_line(self) -> str:
        """一行文字摘要（首末值 + 变化量）。"""
        if self.is_empty or not self.series:
            return "无数据"
        s = self.series[0]
        if not s.values:
            return "无数据"
        first, last = s.values[0], s.values[-1]
        delta = last - first
        sign = "+" if delta > 0 else ""
        unit = s.unit or ""
        return (
            f"{s.name} {_fmt(first)}{unit} → {_fmt(last)}{unit}"
            f"（{sign}{_fmt(delta)}{unit}）"
        )


@dataclass
class ChartRequest:
    """用户一句话的画图请求解析结果。"""

    domain: str = "body"  # body | nutrition | training
    metric: str = "weight"
    metric_label: str = "体重"
    unit: str = "kg"
    start_date: date | None = None
    end_date: date | None = None
    chart_type: str = "line"
    output_mode: str = "inline"  # inline | document | insert
    target_path: Path | None = None
    anchor: str | None = None
    title: str | None = None

    @property
    def is_period(self) -> bool:
        return (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date > self.start_date
        )


# --------------------------------------------------------------------------- #
# 数据聚合
# --------------------------------------------------------------------------- #


def _date_axis(start: date, end: date) -> list[date]:
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(max(days, 0))]


def _short_label(d: date) -> str:
    return f"{d.month:02d}-{d.day:02d}"


def aggregate_body_series(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    metric_type: str = "weight",
) -> tuple[list[str], list[float]]:
    """按日聚合单个身体指标；同一天多来源取均值，缺失日期跳过。"""
    data = invoke_tool(
        query_body_metrics,
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
        metric_type=metric_type,
    )
    buckets: dict[str, list[float]] = {}
    for r in data.get("records", []):
        if r.get("metric_type") != metric_type:
            continue
        buckets.setdefault(r["date"], []).append(float(r["value"]))

    labels: list[str] = []
    values: list[float] = []
    for d in _date_axis(start_date, end_date):
        key = d.isoformat()
        if key in buckets:
            labels.append(_short_label(d))
            values.append(round(sum(buckets[key]) / len(buckets[key]), 2))
    return labels, values


def aggregate_nutrition_series(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    field_name: str = "calories",
    *,
    fill_missing: bool = True,
) -> tuple[list[str], list[float]]:
    """按日聚合饮食营养（缺失日按 0 计，便于柱状图连续）。"""
    data = invoke_tool(
        query_nutrition_logs,
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
    )
    totals = data.get("daily_totals") or {}

    labels: list[str] = []
    values: list[float] = []
    for d in _date_axis(start_date, end_date):
        day = totals.get(d.isoformat())
        if day is None and not fill_missing:
            continue
        labels.append(_short_label(d))
        values.append(round(float((day or {}).get(field_name, 0) or 0), 2))
    return labels, values


def aggregate_training_series(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    field_name: str = "volume_kg",
) -> tuple[list[str], list[float]]:
    """按日聚合训练量（次数 / 容量 / 组数 / 时长 / 消耗）。"""
    data = invoke_tool(
        query_training_logs,
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
    )
    buckets: dict[str, dict[str, float]] = {}
    for s in data.get("sessions", []):
        day = buckets.setdefault(
            s["date"],
            {"sessions": 0.0, "volume_kg": 0.0, "sets": 0.0, "duration_min": 0.0, "calories": 0.0},
        )
        day["sessions"] += 1
        day["volume_kg"] += float(s.get("total_volume_kg") or 0)
        day["sets"] += float(s.get("total_sets") or 0)
        day["duration_min"] += float(s.get("duration_minutes") or 0)
        day["calories"] += float(s.get("calories") or 0)

    labels: list[str] = []
    values: list[float] = []
    for d in _date_axis(start_date, end_date):
        day = buckets.get(d.isoformat())
        if day is None:
            continue
        labels.append(_short_label(d))
        values.append(round(day.get(field_name, 0.0), 2))
    return labels, values


def _downsample(
    labels: list[str], values: list[float], max_points: int
) -> tuple[list[str], list[float]]:
    """等距抽样，始终保留首尾两点。"""
    if max_points <= 0 or len(labels) <= max_points:
        return labels, values
    step = len(labels) / max_points
    picked: list[int] = []
    i = 0.0
    while int(i) < len(labels):
        picked.append(int(i))
        i += step
    if picked[-1] != len(labels) - 1:
        picked.append(len(labels) - 1)
    return [labels[i] for i in picked], [values[i] for i in picked]


# --------------------------------------------------------------------------- #
# 构建图表
# --------------------------------------------------------------------------- #


@tool
def build_body_metric_chart(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    metric_type: str = "weight",
    *,
    chart_type: str = "line",
    title: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> ChartSpec:
    """构建单个身体指标的统计图（默认折线图）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        metric_type: 指标类型，如 weight / bodyfat / weist。
        chart_type: line（折线）或 bar（柱状）。
        title: 图表标题，默认按指标与区间生成。
        max_points: 最大数据点，超出后等距抽样（保留首尾）。
    """
    label = BODY_METRIC_LABELS.get(metric_type, metric_type)
    unit = BODY_METRIC_UNITS.get(metric_type, "")
    labels, values = aggregate_body_series(session, user_id, start_date, end_date, metric_type)
    labels, values = _downsample(labels, values, max_points)

    notes: list[str] = []
    if labels and len(labels) < (end_date - start_date).days + 1:
        notes.append("缺失日期已跳过（当天无记录）。")

    return ChartSpec(
        title=title or f"{label}趋势（{start_date.isoformat()} ~ {end_date.isoformat()}）",
        x_labels=labels,
        series=[ChartSeries(name=label, values=values, unit=unit)],
        y_label=f"{label} ({unit})" if unit else label,
        chart_type=chart_type,
        domain="body",
        metric=metric_type,
        start_date=start_date,
        end_date=end_date,
        notes=notes,
    )


@tool
def build_nutrition_chart(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    field_name: str = "calories",
    *,
    chart_type: str = "bar",
    title: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> ChartSpec:
    """构建饮食营养统计图（默认柱状图，如每日热量 / 蛋白）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        field_name: 营养字段，如 calories / protein_g / carbs_g / fat_g。
        chart_type: line（折线）或 bar（柱状）。
        title: 图表标题，默认按字段与区间生成。
        max_points: 最大数据点。
    """
    label, unit = NUTRITION_FIELD_LABELS.get(field_name, (field_name, ""))
    labels, values = aggregate_nutrition_series(session, user_id, start_date, end_date, field_name)
    labels, values = _downsample(labels, values, max_points)
    return ChartSpec(
        title=title or f"{label}摄入趋势（{start_date.isoformat()} ~ {end_date.isoformat()}）",
        x_labels=labels,
        series=[ChartSeries(name=label, values=values, unit=unit)],
        y_label=f"{label} ({unit})" if unit else label,
        chart_type=chart_type,
        domain="nutrition",
        metric=field_name,
        start_date=start_date,
        end_date=end_date,
    )


@tool
def build_training_chart(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    field_name: str = "volume_kg",
    *,
    chart_type: str = "bar",
    title: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> ChartSpec:
    """构建训练统计量图（默认柱状图，如总容量 / 组数 / 时长）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        field_name: 训练字段，如 volume_kg / sets / duration_min / sessions。
        chart_type: line（折线）或 bar（柱状）。
        title: 图表标题，默认按字段与区间生成。
        max_points: 最大数据点。
    """
    label, unit = TRAINING_FIELD_LABELS.get(field_name, (field_name, ""))
    labels, values = aggregate_training_series(session, user_id, start_date, end_date, field_name)
    labels, values = _downsample(labels, values, max_points)
    return ChartSpec(
        title=title or f"{label}趋势（{start_date.isoformat()} ~ {end_date.isoformat()}）",
        x_labels=labels,
        series=[ChartSeries(name=label, values=values, unit=unit)],
        y_label=f"{label} ({unit})" if unit else label,
        chart_type=chart_type,
        domain="training",
        metric=field_name,
        start_date=start_date,
        end_date=end_date,
    )


def build_chart(
    session: Session,
    user_id: int,
    request: ChartRequest,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> ChartSpec:
    """按 ChartRequest 构建图表（统一入口）。"""
    start = request.start_date or date.today() - timedelta(days=6)
    end = request.end_date or date.today()
    builders = {
        "body": lambda: invoke_tool(
            build_body_metric_chart, session, user_id, start_date=start, end_date=end,
            metric_type=request.metric, chart_type=request.chart_type,
            title=request.title, max_points=max_points,
        ),
        "nutrition": lambda: invoke_tool(
            build_nutrition_chart, session, user_id, start_date=start, end_date=end,
            field_name=request.metric, chart_type=request.chart_type,
            title=request.title, max_points=max_points,
        ),
        "training": lambda: invoke_tool(
            build_training_chart, session, user_id, start_date=start, end_date=end,
            field_name=request.metric, chart_type=request.chart_type,
            title=request.title, max_points=max_points,
        ),
    }
    builder = builders.get(request.domain, builders["body"])
    return builder()


@tool
def generate_chart(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    domain: str,
    metric: str,
    start_date: date,
    end_date: date,
    chart_type: str = "line",
    output_mode: str = "inline",
    target_path: str | None = None,
    anchor: str | None = None,
    title: str | None = None,
) -> dict:
    """生成身体 / 饮食 / 训练数据的 Mermaid 统计图，并按需内联 / 存文档 / 插入文档。

    这是统计图能力对 Agent / 图节点统一的入口：先按 (domain, metric, 日期区间)
    构建图表，再按 output_mode 决定输出形态（对话内联、独立文档、插入已有文档）。
    插入已有文档时会检测是否已存在同指标趋势图，避免重复插入。

    Args:
        domain: 数据域，body / nutrition / training。
        metric: 指标名，如 weight / calories / volume_kg。
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        chart_type: line（折线）或 bar（柱状）。
        output_mode: inline（对话内联）/ document（生成独立文档）/ insert（插入已有文档）。
        target_path: output_mode=insert 时的目标文档路径。
        anchor: insert 时插入到的小节标题（如「## 身体数据趋势」）；为空追加到末尾。
        title: 图表标题，默认按指标与区间生成。

    Returns:
        结构化结果：is_empty / point_count / markdown / output_mode / path /
        duplicate / message 等，便于图节点直接据此组织回复。
    """
    _, metric_label, unit = resolve_metric(metric)
    request = ChartRequest(
        domain=domain,
        metric=metric,
        metric_label=metric_label,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
        chart_type=chart_type,
        output_mode=output_mode,
        target_path=Path(target_path) if target_path else None,
        anchor=anchor,
        title=title,
    )

    chart = build_chart(session, user_id, request)
    markdown = chart.to_markdown()
    result: dict = {
        "is_empty": chart.is_empty,
        "point_count": chart.point_count,
        "chart_type": chart.chart_type,
        "metric_label": metric_label,
        "summary_line": chart.summary_line(),
        "markdown": markdown,
        "output_mode": output_mode,
        "path": None,
        "duplicate": False,
        "draw_skipped": False,
        "message": None,
    }

    if chart.is_empty:
        result["draw_skipped"] = True
        result["message"] = (
            f"{chart.date_range_label or '所选区间'} 内没有{metric_label}记录，"
            f"无法绘制{'折线图' if chart_type == 'line' else '柱状图'}。"
        )
        return result

    if chart.chart_type == "line" and chart.point_count < MIN_LINE_POINTS:
        result["draw_skipped"] = True
        result["message"] = (
            f"{chart.date_range_label or '所选区间'} 内{metric_label}只有 "
            f"{chart.point_count} 个数据点，至少需要 {MIN_LINE_POINTS} 个才能画折线图。"
            f"\n\n{chart.to_table()}"
        )
        return result

    if output_mode == "insert" and request.target_path:
        if _document_has_metric_chart(request.target_path, metric_label):
            result["duplicate"] = True
            result["message"] = (
                f"文档 {request.target_path} 中已包含{metric_label}趋势图，未重复插入。"
                f"如需其它指标请说明，例如「体脂趋势图」。"
            )
            return result
        path = invoke_tool(
            insert_chart_into_document, session, user_id,
            path=str(request.target_path), markdown=markdown, anchor=request.anchor,
        )
        result["path"] = path
        result["message"] = f"已把统计图插入文档：{path}"
        return result

    if output_mode == "document":
        path = write_chart_document(chart, get_settings().chart_output_dir)
        result["path"] = str(path)
        result["message"] = f"已生成统计图文档：{path}"
        return result

    result["message"] = chart.summary_line()
    return result


@tool
def build_body_trend_charts(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    start_date: date,
    end_date: date,
    metrics: list[str] | None = None,
    *,
    max_charts: int = 4,
    max_points: int = DEFAULT_MAX_POINTS,
) -> list[ChartSpec]:
    """周期报表用：为身体指标批量生成折线图（单点指标会被跳过）。

    - 未指定 metrics 时：优先 weight / bodyfat，再用其余有数据的指标补足至 max_charts；
    - 只保留数据点 >= 2 的指标（单点画不出折线）。

    Args:
        start_date: 起始日期（含），ISO 格式 YYYY-MM-DD。
        end_date: 结束日期（含），ISO 格式 YYYY-MM-DD。
        metrics: 指定要绘制的指标列表；为空则自动挑选有数据的指标。
        max_charts: 最多生成的图表数量。
        max_points: 单图最大数据点。
    """
    wanted = list(metrics) if metrics else list(PRIORITY_BODY_METRICS)
    if not metrics:
        try:
            available = available_body_metrics(session, user_id, start_date, end_date)
        except Exception as exc:  # pragma: no cover - 防御性
            logger.warning("查询可用身体指标失败: %s", exc)
            available = []
        for m in available:
            if m not in wanted:
                wanted.append(m)

    charts: list[ChartSpec] = []
    for metric in wanted:
        if len(charts) >= max_charts:
            break
        chart = invoke_tool(
            build_body_metric_chart, session, user_id, start_date=start_date,
            end_date=end_date, metric_type=metric, max_points=max_points,
        )
        if chart.point_count >= MIN_LINE_POINTS:
            charts.append(chart)
    return charts


def available_body_metrics(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
) -> list[str]:
    """返回区间内按数据量降序排列的身体指标类型。"""
    data = invoke_tool(
        query_body_metrics,
        session,
        user_id,
        start_date=start_date,
        end_date=end_date,
    )
    counter: dict[str, int] = {}
    for r in data.get("records", []):
        key = r.get("metric_type")
        if key:
            counter[key] = counter.get(key, 0) + 1
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered]


# --------------------------------------------------------------------------- #
# 请求解析
# --------------------------------------------------------------------------- #

_BODY_METRIC_KEYWORDS = {
    "体重": "weight",
    "体脂": "bodyfat",
    "腰围": "weist",
    "胸围": "chest",
    "臀围": "bot",
    "肩围": "shoulder",
    "颈围": "neck",
    "腿围": "leg_left",
    "臂围": "arm_left",
}
_NUTRITION_KEYWORDS = {
    "热量": "calories",
    "卡路里": "calories",
    "大卡": "calories",
    "蛋白": "protein_g",
    "碳水": "carbs_g",
    "脂肪": "fat_g",
}
_TRAINING_KEYWORDS = {
    "容量": "volume_kg",
    "组数": "sets",
    "时长": "duration_min",
    "训练次数": "sessions",
    "训练量": "volume_kg",
}


def is_chart_request(text: str) -> bool:
    return any(k in text for k in CHART_REQUEST_KEYWORDS)


def resolve_metric(metric: str) -> tuple[str, str, str]:
    """由指标名反查 (domain, 中文名, 单位)；未知指标兜底为体重。"""
    if metric in BODY_METRIC_LABELS:
        return "body", BODY_METRIC_LABELS[metric], BODY_METRIC_UNITS.get(metric, "")
    if metric in NUTRITION_FIELD_LABELS:
        label, unit = NUTRITION_FIELD_LABELS[metric]
        return "nutrition", label, unit
    if metric in TRAINING_FIELD_LABELS:
        label, unit = TRAINING_FIELD_LABELS[metric]
        return "training", label, unit
    return "body", BODY_METRIC_LABELS["weight"], BODY_METRIC_UNITS["weight"]


def parse_chart_request(
    message: str,
    today: date | None = None,
    *,
    default_days: int = 7,
    reports_dir: str | Path | None = None,
) -> ChartRequest:
    """从用户消息解析画图请求（指标 / 日期范围 / 图表类型 / 输出方式）。"""
    today = today or date.today()
    text = message.strip()

    domain, metric, label, unit = _infer_metric(text)
    chart_type = _infer_chart_type(text, domain)
    start, end = _infer_date_range(text, today, default_days=default_days)
    output_mode, target_path, anchor = _infer_output_mode(text, reports_dir=reports_dir)

    return ChartRequest(
        domain=domain,
        metric=metric,
        metric_label=label,
        unit=unit,
        start_date=start,
        end_date=end,
        chart_type=chart_type,
        output_mode=output_mode,
        target_path=target_path,
        anchor=anchor,
    )


def _infer_metric(text: str) -> tuple[str, str, str, str]:
    for kw, metric in _TRAINING_KEYWORDS.items():
        if kw in text:
            label, unit = TRAINING_FIELD_LABELS.get(metric, (metric, ""))
            return "training", metric, label, unit
    for kw, metric in _NUTRITION_KEYWORDS.items():
        if kw in text:
            label, unit = NUTRITION_FIELD_LABELS.get(metric, (metric, ""))
            return "nutrition", metric, label, unit
    for kw, metric in _BODY_METRIC_KEYWORDS.items():
        if kw in text:
            return "body", metric, BODY_METRIC_LABELS.get(metric, metric), BODY_METRIC_UNITS.get(
                metric, ""
            )
    # 训练相关兜底：出现「训练/容量」等但没命中具体词
    if any(k in text for k in ("训练", "深蹲", "卧推", "硬拉")):
        return "training", "volume_kg", *TRAINING_FIELD_LABELS["volume_kg"]
    return "body", "weight", BODY_METRIC_LABELS["weight"], BODY_METRIC_UNITS["weight"]


def _infer_chart_type(text: str, domain: str) -> str:
    for kw, chart_type in CHART_TYPE_KEYWORDS.items():
        if kw in text:
            return chart_type
    # 饮食/训练按日汇总默认柱状图，身体趋势默认折线图
    return "line" if domain == "body" else "bar"


def _infer_date_range(text: str, today: date, *, default_days: int) -> tuple[date, date]:
    """解析日期区间：显式区间 > 最近N天 > 单日 > 默认最近 default_days 天。"""
    from myfitness.agents.tools.query_planner import parse_date_range_text

    start, end = parse_date_range_text(text, today)
    if start and end:
        return start, end
    if start:
        return start, start

    if m := _RECENT_DAYS_RE.search(text):
        days = max(int(m.group(1)), 1)
        return today - timedelta(days=days - 1), today

    return today - timedelta(days=default_days - 1), today


def _infer_output_mode(
    text: str, *, reports_dir: str | Path | None
) -> tuple[str, Path | None, str | None]:
    """判断输出方式：inline（对话内联）/ document（生成文档）/ insert（插入已有文档）。"""
    wants_insert = any(k in text for k in INSERT_DOC_KEYWORDS)
    mentions_doc = any(k in text for k in _DOCUMENT_KEYWORDS)

    if wants_insert:
        target = _resolve_target_document(text, reports_dir=reports_dir)
        if target is not None:
            anchor = _infer_anchor(text)
            return "insert", target, anchor
        # 没定位到目标文档也要求插入 → 退化成生成文档
        return "document", None, None

    if mentions_doc:
        return "document", _resolve_target_document(text, reports_dir=reports_dir), None

    return "inline", None, None


def _resolve_target_document(text: str, *, reports_dir: str | Path | None) -> Path | None:
    """从消息中推断目标文档路径（显式 .md 路径 或 reports 下的某日报表）。"""
    from myfitness.agents.tools.query_planner import parse_single_date

    if m := re.search(r"([A-Za-z]:\\[^\s]+\.md|[\w./-]+\.md)", text):
        return Path(m.group(1))

    if reports_dir is None:
        return None
    d = parse_single_date(text)
    if d is None:
        return None
    base = Path(reports_dir)
    for candidate in (base / f"{d.isoformat()}.md", base / f"{d.isoformat()}_{d.isoformat()}.md"):
        if candidate.exists():
            return candidate
    return base / f"{d.isoformat()}.md"


def _infer_anchor(text: str) -> str | None:
    """解析插入位置：「插入到 ## 趋势图 下」中的标题；否则返回 None（追加到末尾）。"""
    m = re.search(r"(?:插入|加到|加进|追加|放进|补充到)\s*(?:到)?\s*(#{1,6}\s*[^\s，。]+)", text)
    if m:
        return m.group(1).strip()
    pattern = (
        r"(?:插入|加到|加进|追加|放进|补充到)\s*(?:到)?\s*「?"
        r"([^」\s，。]{2,12}?)」?\s*(?:下面|下方|之后|后面)"
    )
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return None


# --------------------------------------------------------------------------- #
# 文档输出
# --------------------------------------------------------------------------- #


def render_chart_document(chart: ChartSpec, *, generated_at: str | None = None) -> str:
    """生成一份独立的图表 Markdown 文档。"""
    from datetime import UTC, datetime

    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"# {chart.title}",
        "",
        f"> 生成时间：{stamp}",
    ]
    if chart.date_range_label:
        parts.append(f"> 数据范围：{chart.date_range_label}")
    parts += ["", chart.to_markdown(include_table=True, heading_level=2)]
    return "\n".join(parts) + "\n"


def write_chart_document(
    chart: ChartSpec,
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """把图表写成独立 Markdown 文档，返回文件路径。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = filename or default_chart_filename(chart)
    if not name.endswith(".md"):
        name += ".md"
    path = out / name
    path.write_text(render_chart_document(chart), encoding="utf-8")
    return path


def default_chart_filename(chart: ChartSpec) -> str:
    span = (
        f"{chart.start_date.isoformat()}_{chart.end_date.isoformat()}"
        if chart.start_date and chart.end_date
        else "recent"
    )
    return f"chart-{chart.domain}-{chart.metric}-{span}.md"


@tool
def insert_chart_into_document(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    path: str,
    markdown: str,
    anchor: str | None = None,
    *,
    create_if_missing: bool = True,
) -> str:
    """把图表 Markdown 插入已有文档，返回文档路径。

    - anchor=None：追加到文件末尾；
    - anchor 为标题文本或 `## xxx`：插入到该小节末尾（下一个同级/更高级标题之前）；
    - 找不到 anchor：追加到末尾并补上该标题（anchor 为标题时）；
    - 文件不存在且 create_if_missing：新建文档。

    Args:
        path: 目标 Markdown 文档路径。
        markdown: 要插入的图表 Markdown 片段。
        anchor: 插入位置的小节标题；为空则追加到末尾。
    """
    target = Path(path)
    if not target.exists():
        if not create_if_missing:
            raise FileNotFoundError(str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        return str(target)

    content = target.read_text(encoding="utf-8")
    lines = content.splitlines()

    if anchor:
        index = _find_anchor_index(lines, anchor)
        if index is not None:
            end_index = _section_end_index(lines, index)
            block = markdown.rstrip().splitlines()
            new_lines = lines[:end_index] + [""] + block + lines[end_index:]
            target.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
            return str(target)
        # 未找到锚点 → 以标题形式补在末尾
        heading = anchor if anchor.lstrip().startswith("#") else f"## {anchor}"
        lines = lines + ["", heading, "", markdown.strip()]

    else:
        lines = lines + ["", markdown.strip()]

    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(target)


def _find_anchor_index(lines: list[str], anchor: str) -> int | None:
    needle = anchor.lstrip().lstrip("#").strip().lower()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            continue
        title = line.lstrip().lstrip("#").strip().lower()
        if title == needle or needle in title:
            return i
    # 也允许非标题行作为锚点
    for i, line in enumerate(lines):
        if needle and needle in line.lower():
            return i
    return None


def _section_end_index(lines: list[str], heading_index: int) -> int:
    """返回小节结束位置（下一个同级或更高级标题行，或文件末尾）。"""
    level = len(lines[heading_index]) - len(lines[heading_index].lstrip("#"))
    for i in range(heading_index + 1, len(lines)):
        line = lines[i]
        if line.lstrip().startswith("#"):
            cur = len(line) - len(line.lstrip("#"))
            if cur <= level:
                return i
    return len(lines)


def _document_has_metric_chart(path: Path, metric_label: str) -> bool:
    """目标文档中是否已存在该指标的趋势图（避免重复插入）。"""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return "xychart-beta" in content and f"{metric_label}趋势（" in content


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


def _fmt(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("[", "(").replace("]", ")")


def _pad_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """给 y 轴留 15% 余量；数据非负时下界不越到负数（热量 / 训练容量等）。"""
    low, high = bounds
    non_negative = low >= 0
    if low == high:
        low, high = low - 1, high + 1
    else:
        pad = (high - low) * 0.15
        low, high = low - pad, high + pad
    if non_negative:
        low = max(0.0, low)
    if low == high:  # 兜底：保证区间非空
        high = low + 1
    return round(low, 2), round(high, 2)
