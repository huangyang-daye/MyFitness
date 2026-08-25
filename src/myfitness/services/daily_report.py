"""日报生成工作流 — 同步 → 分析 → 写入 DB / 文件。"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from myfitness.agents.tools.query_format import format_training_sessions_detail
from myfitness.agents.tools.query_tools import (
    query_body_metrics,
    query_nutrition_logs,
    query_training_logs,
)
from myfitness.agents.body_monitor import run_body_agent
from myfitness.agents.fitness_planner import run_fitness_agent
from myfitness.agents.nutritionist import run_nutrition_agent
from myfitness.agents.summary import build_rule_based_summary, run_summary_agent
from myfitness.config import get_settings
from myfitness.db.repositories.reports import DailyReportRepository
from myfitness.db.session import get_or_create_default_user
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.constants import DISCLAIMER
from myfitness.schemas.state import Intent
from myfitness.services.context_loader import load_context_snapshot
from myfitness.sync.orchestrator import run_sync

logger = logging.getLogger(__name__)


def run_daily_report(
    session: Session,
    user_id: int,
    report_date: date | None = None,
    *,
    sync_first: bool = True,
    lookback_days: int = 30,
) -> dict:
    """生成单日日报：可选先同步，再跑三 Agent + Summary，持久化并写文件。"""
    get_or_create_default_user(session, user_id)
    settings = get_settings()
    report_date = report_date or (date.today() - timedelta(days=1))

    sync_result: dict | None = None
    if sync_first:
        try:
            sync_result = run_sync(
                session,
                user_id,
                start_date=report_date,
                end_date=report_date,
                days=1,
            )
        except Exception as exc:
            logger.warning("日报同步失败，继续使用已有数据: %s", exc)
            sync_result = {"status": "failed", "errors": [str(exc)]}

    context = load_context_snapshot(
        session,
        user_id,
        end_date=report_date,
        lookback_days=lookback_days,
    )

    outputs = AgentOutputs(
        body=run_body_agent(context, analysis_date=report_date),
        nutrition=run_nutrition_agent(context, analysis_date=report_date),
        fitness=run_fitness_agent(context, analysis_date=report_date),
    )
    summary = run_summary_agent(
        outputs,
        context,
        Intent.TREND_ANALYSIS,
        user_message=f"生成 {report_date.isoformat()} 日报",
        output_type="daily_report",
    )

    content_md = format_daily_report_md(
        report_date=report_date,
        context=context,
        summary_content=summary.content_md,
        sync_result=sync_result,
        body_metrics=_load_report_day_body(session, user_id, report_date),
        nutrition_logs=_load_report_day_nutrition(session, user_id, report_date),
        training_sessions=_load_report_day_training(session, user_id, report_date),
    )

    repo = DailyReportRepository(session, user_id)
    report = repo.upsert(
        report_date,
        content_md,
        agent_outputs=outputs.model_dump(mode="json"),
    )

    file_path = write_report_file(content_md, report_date, settings.daily_report_output_dir)

    return {
        "report_date": report_date.isoformat(),
        "report_id": report.id,
        "file_path": str(file_path) if file_path else None,
        "sync": sync_result,
        "content_md": content_md,
    }


def format_daily_report_md(
    report_date: date,
    context,
    summary_content: str,
    sync_result: dict | None = None,
    body_metrics: list[dict] | None = None,
    nutrition_logs: dict | None = None,
    training_sessions: list[dict] | None = None,
) -> str:
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

MEAL_LABELS = {
    "breakfast": "早餐",
    "lunch": "午餐",
    "dinner": "晚餐",
    "snack": "加餐",
    "other": "其他",
}

MEAL_ORDER = {"breakfast": 0, "lunch": 1, "dinner": 2, "snack": 3, "other": 4}


def format_body_metrics_detail(records: list[dict]) -> str:
    if not records:
        return "报告日无身体数据记录。"

    lines = [
        "| 指标 | 数值 | 来源 |",
        "|---|---:|---|",
    ]
    for record in sorted(records, key=lambda r: (r.get("metric_type") or "", r.get("source") or "")):
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


def write_report_file(content_md: str, report_date: date, output_dir: str) -> Path | None:
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{report_date.isoformat()}.md"
        path.write_text(content_md, encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning("写入日报文件失败: %s", exc)
        return None


def build_quick_report_preview(session: Session, user_id: int, report_date: date) -> str:
    """对话中即时预览（不写入）。"""
    context = load_context_snapshot(session, user_id, end_date=report_date, lookback_days=7)
    outputs = AgentOutputs(
        body=run_body_agent(context, analysis_date=report_date),
        nutrition=run_nutrition_agent(context, analysis_date=report_date),
        fitness=run_fitness_agent(context, analysis_date=report_date),
    )
    return build_rule_based_summary(outputs, context, Intent.TREND_ANALYSIS)


def _load_report_day_body(session: Session, user_id: int, report_date: date) -> list[dict]:
    result = query_body_metrics(session, user_id, report_date, report_date)
    return result.get("records") or []


def _load_report_day_nutrition(session: Session, user_id: int, report_date: date) -> dict:
    return query_nutrition_logs(session, user_id, report_date, report_date)


def _load_report_day_training(session: Session, user_id: int, report_date: date) -> list[dict]:
    result = query_training_logs(session, user_id, report_date, report_date)
    return result.get("sessions") or []
