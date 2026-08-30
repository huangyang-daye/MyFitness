"""周期报表 — 可指定日期区间的健康报告（单日退化为日报）。

- 区间仅 1 天：输出与原「日报」完全一致的格式；
- 区间 > 1 天：在日报结构基础上追加「身体数据趋势图」（Mermaid 折线图）、
  每日明细表与区间汇总。
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from myfitness.agents.body_monitor import run_body_agent
from myfitness.agents.fitness_planner import run_fitness_agent
from myfitness.agents.nutritionist import run_nutrition_agent
from myfitness.agents.summary import run_summary_agent
from myfitness.agents.tools.base import invoke_tool
from myfitness.agents.tools.chart_tools import (
    BODY_METRIC_LABELS,
    build_body_trend_charts,
)
from myfitness.agents.tools.query_format import format_training_sessions_detail
from myfitness.agents.tools.query_tools import (
    query_body_metrics,
    query_nutrition_logs,
    query_training_logs,
)
from myfitness.config import get_settings
from myfitness.db.repositories.reports import DailyReportRepository
from myfitness.db.session import get_or_create_default_user
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.constants import DISCLAIMER
from myfitness.schemas.state import ContextSnapshot, Intent
from myfitness.services.context_loader import load_context_snapshot
from myfitness.sync.orchestrator import run_sync

logger = logging.getLogger(__name__)

MEAL_LABELS = {
    "breakfast": "早餐",
    "lunch": "午餐",
    "dinner": "晚餐",
    "snack": "加餐",
    "other": "其他",
}

MEAL_ORDER = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3, "other": 4}


def run_period_report(
    session: Session,
    user_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    sync_first: bool = True,
    lookback_days: int = 30,
    include_charts: bool = True,
    chart_metrics: list[str] | None = None,
) -> dict:
    """生成区间报表：可选先同步，再跑三 Agent + Summary，持久化并写文件。

    start_date == end_date（或只给一个日期）时退化为单日日报。
    """
    start_date, end_date = normalize_period(start_date, end_date)

    sync_result: dict | None = None
    if sync_first:
        try:
            sync_result = run_sync(
                session,
                user_id,
                start_date=start_date,
                end_date=end_date,
                days=(end_date - start_date).days + 1,
            )
        except Exception as exc:
            logger.warning("周期报表同步失败，继续使用已有数据: %s", exc)
            sync_result = {"status": "failed", "errors": [str(exc)]}

    return render_period_report(
        session,
        user_id,
        start_date,
        end_date,
        sync_result=sync_result,
        lookback_days=lookback_days,
        include_charts=include_charts,
        chart_metrics=chart_metrics,
    )


def normalize_period(
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    """规范化日期区间：缺失的一侧用另一侧补齐，都缺失时为昨天。"""
    if start_date and end_date:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        return start_date, end_date
    if start_date:
        return start_date, start_date
    if end_date:
        return end_date, end_date
    yesterday = date.today() - timedelta(days=1)
    return yesterday, yesterday


def render_period_report(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
    *,
    sync_result: dict | None = None,
    lookback_days: int = 30,
    include_charts: bool = True,
    chart_metrics: list[str] | None = None,
) -> dict:
    """渲染并持久化区间报表（不做同步，同步由调用方负责）。"""
    get_or_create_default_user(session, user_id)
    settings = get_settings()
    start_date, end_date = normalize_period(start_date, end_date)
    span_days = (end_date - start_date).days + 1
    is_daily = span_days <= 1

    body_metrics = invoke_tool(
        query_body_metrics, session, user_id, start_date=start_date, end_date=end_date
    ).get("records") or []
    nutrition_logs = invoke_tool(
        query_nutrition_logs, session, user_id, start_date=start_date, end_date=end_date
    )
    training_sessions = (
        invoke_tool(
            query_training_logs, session, user_id, start_date=start_date, end_date=end_date
        ).get("sessions")
        or []
    )

    # 多日报表：把区间查询结果注入上下文，让 Agent 的分析口径与报告区间一致
    # （单日日报保持原有 lookback 口径，避免改变既有输出）
    query_results = (
        {
            "body": {
                "tool": "query_body_metrics",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "count": len(body_metrics),
                "records": body_metrics,
            },
            "nutrition": nutrition_logs,
            "training": {
                "tool": "query_training_logs",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "count": len(training_sessions),
                "sessions": training_sessions,
            },
        }
        if not is_daily
        else None
    )

    context = load_context_snapshot(
        session,
        user_id,
        end_date=end_date,
        lookback_days=max(lookback_days, span_days),
        query_results=query_results,
    )
    # load_context_snapshot 只用 query_results 计算 summary，不回写字段；
    # 这里显式挂上，让 Agent 的「查询范围内」文案与报告区间一致（不影响对话链路）
    if query_results:
        context.query_results = query_results

    outputs = AgentOutputs(
        body=run_body_agent(context, analysis_date=end_date),
        nutrition=run_nutrition_agent(context, analysis_date=end_date),
        fitness=run_fitness_agent(context, analysis_date=end_date),
    )
    range_label = (
        end_date.isoformat() if is_daily else f"{start_date.isoformat()} ~ {end_date.isoformat()}"
    )
    summary = run_summary_agent(
        outputs,
        context,
        Intent.TREND_ANALYSIS,
        user_message=f"生成 {range_label} 健康{'日报' if is_daily else '周期报表'}",
        output_type="daily_report" if is_daily else "period_report",
        # 周期报表自带趋势图与每日明细，摘要里不再罗列查询明细
        include_query_results=is_daily,
    )

    charts = []
    if include_charts and not is_daily:
        try:
            charts = invoke_tool(
                build_body_trend_charts,
                session,
                user_id,
                start_date=start_date,
                end_date=end_date,
                metrics=chart_metrics,
            )
        except Exception as exc:
            logger.warning("生成趋势图失败，报告中省略图表: %s", exc)
            charts = []

    if is_daily:
        content_md = format_daily_report_md(
            report_date=end_date,
            context=context,
            summary_content=summary.content_md,
            sync_result=sync_result,
            body_metrics=body_metrics,
            nutrition_logs=nutrition_logs,
            training_sessions=training_sessions,
        )
    else:
        content_md = format_period_report_md(
            start_date=start_date,
            end_date=end_date,
            context=context,
            summary_content=summary.content_md,
            sync_result=sync_result,
            body_metrics=body_metrics,
            nutrition_logs=nutrition_logs,
            training_sessions=training_sessions,
            charts=charts,
        )

    agent_payload = outputs.model_dump(mode="json")
    agent_payload["period"] = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days": span_days,
        "report_kind": "daily" if is_daily else "period",
        "charts": [c.metric for c in charts],
    }

    repo = DailyReportRepository(session, user_id)
    report = repo.upsert(end_date, content_md, agent_outputs=agent_payload)

    file_path = write_report_file(
        content_md, start_date, end_date, settings.daily_report_output_dir
    )

    return {
        "report_date": end_date.isoformat(),
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "period_days": span_days,
        "report_kind": "daily" if is_daily else "period",
        "report_id": report.id,
        "file_path": str(file_path) if file_path else None,
        "sync": sync_result,
        "content_md": content_md,
        "charts": [{"metric": c.metric, "title": c.title} for c in charts],
    }


# --------------------------------------------------------------------------- #
# 格式化
# --------------------------------------------------------------------------- #


def format_daily_report_md(
    report_date: date,
    context: ContextSnapshot,
    summary_content: str,
    sync_result: dict | None = None,
    body_metrics: list[dict] | None = None,
    nutrition_logs: dict | None = None,
    training_sessions: list[dict] | None = None,
) -> str:
    """单日日报（与历史格式保持一致）。"""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    dr = context.date_range
    gaps = context.data_gaps or []
    banner = "⚠ 部分数据缺失：" + "；".join(gaps) if gaps else "数据完整"
    body_detail = format_body_metrics_detail(body_metrics or [])
    nutrition_detail = format_nutrition_logs_detail(nutrition_logs or {})
    training_detail = format_training_sessions_detail(training_sessions or [])

    sync_line = ""
    if sync_result:
        sync_line = f"\n> 同步状态：{sync_result.get('status', 'unknown')}"

    header = f"""# MyFitness 日报 — {report_date.isoformat()}

> 生成时间：{generated_at}{sync_line}
> 数据覆盖：{dr.start.isoformat()} ~ {dr.end.isoformat()}
> {banner}

---

## 分析摘要

{summary_content}

---

## 原始指标速览

### 身体（报告日）
{body_detail}

### 饮食（报告日）
{nutrition_detail}

### 训练（报告日）
{training_detail}

---

_{DISCLAIMER}_
"""
    return header


def format_period_report_md(
    start_date: date,
    end_date: date,
    context: ContextSnapshot,
    summary_content: str,
    sync_result: dict | None = None,
    body_metrics: list[dict] | None = None,
    nutrition_logs: dict | None = None,
    training_sessions: list[dict] | None = None,
    charts: list | None = None,
) -> str:
    """多日周期报表：摘要 → 身体趋势图 → 每日明细 → 区间汇总 → 训练明细。"""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    span_days = (end_date - start_date).days + 1
    dr = context.date_range
    gaps = context.data_gaps or []
    banner = "⚠ 部分数据缺失：" + "；".join(gaps) if gaps else "数据完整"

    sync_line = ""
    if sync_result:
        sync_line = f"\n> 同步状态：{sync_result.get('status', 'unknown')}"

    charts = list(charts or [])
    if charts:
        chart_sections = "\n\n".join(c.to_markdown(heading_level=3) for c in charts)
        chart_block = f"## 身体数据趋势\n\n{chart_sections}"
    else:
        chart_block = (
            "## 身体数据趋势\n\n"
            "区间内身体指标数据点不足（至少需要同一指标 2 天记录），暂不生成趋势图。"
        )

    body_metrics = body_metrics or []
    nutrition_logs = nutrition_logs or {}
    training_sessions = training_sessions or []

    daily_table = format_daily_breakdown(
        start_date, end_date, body_metrics, nutrition_logs, training_sessions
    )
    highlights = format_period_highlights(
        body_metrics, nutrition_logs, training_sessions, span_days
    )
    training_detail = format_training_period_detail(training_sessions)

    title = f"# MyFitness 周期报表 — {start_date.isoformat()} ~ {end_date.isoformat()}（{span_days} 天）"
    return f"""{title}

> 生成时间：{generated_at}{sync_line}
> 报告区间：{start_date.isoformat()} ~ {end_date.isoformat()}（{span_days} 天）
> 数据覆盖：{dr.start.isoformat()} ~ {dr.end.isoformat()}
> {banner}

---

## 分析摘要

{summary_content}

---

{chart_block}

---

## 每日明细

{daily_table}

---

## 区间汇总

{highlights}

---

## 训练明细

{training_detail}

---

_{DISCLAIMER}_
"""


def format_daily_breakdown(
    start_date: date,
    end_date: date,
    body_metrics: list[dict],
    nutrition_logs: dict,
    training_sessions: list[dict],
) -> str:
    """按日汇总体重 / 体脂 / 热量 / 蛋白 / 训练次数。"""
    body_by_date = _group_body_by_date(body_metrics)
    nutrition_by_date = nutrition_logs.get("daily_totals") or {}
    training_by_date: dict[str, int] = {}
    for s in training_sessions:
        d = s.get("date")
        if d:
            training_by_date[d] = training_by_date.get(d, 0) + 1

    lines = [
        "| 日期 | 体重 (kg) | 体脂率 (%) | 热量 (kcal) | 蛋白质 (g) | 训练次数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    cursor = start_date
    while cursor <= end_date:
        key = cursor.isoformat()
        body = body_by_date.get(key, {})
        ntr = nutrition_by_date.get(key) or {}
        weight = body.get("weight")
        bodyfat = body.get("bodyfat")
        lines.append(
            "| "
            f"{key} | "
            f"{_fmt_cell(weight)} | "
            f"{_fmt_cell(bodyfat)} | "
            f"{_fmt_cell(ntr.get('calories'), default='—')} | "
            f"{_fmt_cell(ntr.get('protein_g'), default='—')} | "
            f"{training_by_date.get(key, 0)} |"
        )
        cursor += timedelta(days=1)
    return "\n".join(lines)


def format_period_highlights(
    body_metrics: list[dict],
    nutrition_logs: dict,
    training_sessions: list[dict],
    span_days: int,
) -> str:
    """区间首尾对比与均值。"""
    lines: list[str] = []
    body_by_date = _group_body_by_date(body_metrics)

    for metric in ("weight", "bodyfat"):
        points = [
            (d, values[metric])
            for d, values in sorted(body_by_date.items())
            if metric in values
        ]
        if len(points) >= 2:
            label = BODY_METRIC_LABELS.get(metric, metric)
            unit = "%" if metric == "bodyfat" else "kg"
            delta = points[-1][1] - points[0][1]
            sign = "+" if delta > 0 else ""
            lines.append(
                f"- {label}：{_format_number(points[0][1])} → {_format_number(points[-1][1])} "
                f"{unit}（{sign}{_format_number(delta)} {unit}，{len(points)} 个记录日）"
            )

    totals = nutrition_logs.get("daily_totals") or {}
    if totals:
        days = len(totals)
        calories = sum(float(v.get("calories", 0) or 0) for v in totals.values())
        protein = sum(float(v.get("protein_g", 0) or 0) for v in totals.values())
        lines.append(
            f"- 饮食：{days} 天有记录，日均 {_format_number(calories / days)} kcal，"
            f"蛋白质日均 {_format_number(protein / days)} g"
        )
    else:
        lines.append("- 饮食：区间内无饮食记录")

    if training_sessions:
        volume = sum(float(s.get("total_volume_kg") or 0) for s in training_sessions)
        lines.append(
            f"- 训练：{len(training_sessions)} 次，总容量 {_format_number(volume)} kg"
            + (f"，平均 {_format_number(len(training_sessions) / span_days * 7)} 次/周"
               if span_days >= 7 else "")
        )
    else:
        lines.append("- 训练：区间内无训练记录")

    return "\n".join(lines)


def format_training_period_detail(training_sessions: list[dict]) -> str:
    """按日列出训练概要（避免长报告塞满每组细节）。"""
    if not training_sessions:
        return "区间内无训练记录。"

    by_date: dict[str, list[dict]] = {}
    for s in training_sessions:
        by_date.setdefault(s.get("date") or "未知日期", []).append(s)

    lines: list[str] = []
    for day in sorted(by_date):
        lines.append(f"**{day}**")
        for s in by_date[day]:
            bits = [s.get("title") or "训练"]
            if s.get("duration_minutes"):
                bits.append(f"{_format_number(s['duration_minutes'])} min")
            if s.get("total_volume_kg"):
                bits.append(f"{_format_number(s['total_volume_kg'])} kg")
            if s.get("total_sets"):
                bits.append(f"{s['total_sets']} 组")
            lines.append("- " + " · ".join(bits))
    return "\n".join(lines)


def format_body_metrics_detail(records: list[dict]) -> str:
    if not records:
        return "报告日无身体数据记录。"

    lines = [
        "| 指标 | 数值 | 来源 |",
        "|---|---:|---|",
    ]
    for record in sorted(
        records,
        key=lambda r: (r.get("metric_type") or "", r.get("source") or ""),
    ):
        metric_type = record.get("metric_type") or "unknown"
        label = BODY_METRIC_LABELS.get(metric_type, metric_type)
        value = _format_number(record.get("value"))
        unit = record.get("unit") or ""
        source = record.get("source") or "—"
        lines.append(f"| {label} | {value} {unit} | {source} |")
    return "\n".join(lines)


def format_nutrition_logs_detail(data: dict) -> str:
    entries = data.get("entries") or []
    totals = _report_day_totals(data)
    lines = [
        "| 合计 | 热量 | 蛋白质 | 碳水 | 脂肪 |",
        "|---|---:|---:|---:|---:|",
        (
            "| 当日合计 | "
            f"{_format_number(totals.get('calories'))} kcal | "
            f"{_format_number(totals.get('protein_g'))} g | "
            f"{_format_number(totals.get('carbs_g'))} g | "
            f"{_format_number(totals.get('fat_g'))} g |"
        ),
    ]

    if not entries:
        lines.append("")
        lines.append("报告日无饮食明细记录。")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| 餐次 | 食物 | 份量 | 热量 | 蛋白质 | 碳水 | 脂肪 | 来源 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for entry in sorted(entries, key=_nutrition_sort_key):
        ntr = entry.get("nutrients") or {}
        meal = MEAL_LABELS.get(entry.get("meal_type"), entry.get("meal_type") or "其他")
        amount = _format_number(entry.get("amount"))
        unit = entry.get("unit") or ""
        lines.append(
            "| "
            f"{meal} | "
            f"{entry.get('food_name') or '未知食物'} | "
            f"{amount} {unit} | "
            f"{_format_number(ntr.get('calories'))} kcal | "
            f"{_format_number(ntr.get('protein_g'))} g | "
            f"{_format_number(ntr.get('carbs_g'))} g | "
            f"{_format_number(ntr.get('fat_g'))} g | "
            f"{entry.get('source') or '—'} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 文件输出
# --------------------------------------------------------------------------- #


def write_report_file(
    content_md: str,
    start_date: date,
    end_date: date,
    output_dir: str,
) -> Path | None:
    """单日 → `YYYY-MM-DD.md`；多日 → `YYYY-MM-DD_YYYY-MM-DD.md`。"""
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = (
            f"{start_date.isoformat()}.md"
            if start_date == end_date
            else f"{start_date.isoformat()}_{end_date.isoformat()}.md"
        )
        path = out / name
        path.write_text(content_md, encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning("写入报表文件失败: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #


def _group_body_by_date(records: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for r in records:
        day = grouped.setdefault(r.get("date"), {})
        metric = r.get("metric_type")
        if metric:
            day[metric] = float(r.get("value"))
    return grouped


def _report_day_totals(data: dict) -> dict[str, float]:
    daily_totals = data.get("daily_totals") or {}
    if daily_totals:
        return next(iter(daily_totals.values()))
    return {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}


def _nutrition_sort_key(entry: dict) -> tuple[int, str]:
    meal_type = entry.get("meal_type") or "other"
    return (MEAL_ORDER.get(meal_type, MEAL_ORDER["other"]), entry.get("food_name") or "")


def _format_number(value: object) -> str:
    if value is None:
        return "0"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _fmt_cell(value: object, *, default: str = "—") -> str:
    if value is None or value == "":
        return default
    return _format_number(value)
