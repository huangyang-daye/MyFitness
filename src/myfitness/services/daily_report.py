"""日报生成工作流 — 单日场景入口。

周期报表（可指定日期区间）见 `services/period_report.py`；
本模块保留 `run_daily_report` 作为单日入口，内部委托给周期报表渲染器，
单日输出格式与历史完全一致。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from myfitness.agents.body_monitor import run_body_agent
from myfitness.agents.fitness_planner import run_fitness_agent
from myfitness.agents.nutritionist import run_nutrition_agent
from myfitness.agents.summary import build_rule_based_summary
from myfitness.agents.tools.chart_tools import BODY_METRIC_LABELS
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import Intent
from myfitness.services.context_loader import load_context_snapshot
from myfitness.services.period_report import (
    MEAL_LABELS,
    MEAL_ORDER,
    format_body_metrics_detail,
    format_daily_breakdown,
    format_daily_report_md,
    format_nutrition_logs_detail,
    format_period_highlights,
    format_period_report_md,
    normalize_period,
    render_period_report,
    run_period_report,
    write_report_file,
)
from myfitness.sync.orchestrator import run_sync

logger = logging.getLogger(__name__)

__all__ = [
    "BODY_METRIC_LABELS",
    "MEAL_LABELS",
    "MEAL_ORDER",
    "build_quick_report_preview",
    "format_body_metrics_detail",
    "format_daily_breakdown",
    "format_daily_report_md",
    "format_nutrition_logs_detail",
    "format_period_highlights",
    "format_period_report_md",
    "normalize_period",
    "render_period_report",
    "run_daily_report",
    "run_period_report",
    "write_report_file",
]


def run_daily_report(
    session: Session,
    user_id: int,
    report_date: date | None = None,
    *,
    sync_first: bool = True,
    lookback_days: int = 30,
) -> dict:
    """生成单日日报：可选先同步，再跑三 Agent + Summary，持久化并写文件。

    等价于区间退化为 1 天的周期报表（`run_period_report`）。
    """
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

    result = render_period_report(
        session,
        user_id,
        report_date,
        report_date,
        sync_result=sync_result,
        lookback_days=lookback_days,
    )
    result["sync"] = sync_result
    return result


def build_quick_report_preview(session: Session, user_id: int, report_date: date) -> str:
    """对话中即时预览（不写入）。"""
    context = load_context_snapshot(session, user_id, end_date=report_date, lookback_days=7)
    outputs = AgentOutputs(
        body=run_body_agent(context, analysis_date=report_date),
        nutrition=run_nutrition_agent(context, analysis_date=report_date),
        fitness=run_fitness_agent(context, analysis_date=report_date),
    )
    return build_rule_based_summary(outputs, context, Intent.TREND_ANALYSIS)
