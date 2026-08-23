"""日报生成工作流 — 同步 → 分析 → 写入 DB / 文件。"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from myfitness.agents.tools.query_format import format_training_sessions_detail
from myfitness.agents.tools.query_tools import query_training_logs
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
    training_sessions: list[dict] | None = None,
) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    dr = context.date_range
    gaps = context.data_gaps or []
    banner = "⚠ 部分数据缺失：" + "；".join(gaps) if gaps else "数据完整"
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

### 身体
- 最新体重：{context.body_metrics_summary.get('latest_weight_kg') or '—'} kg
- 最新体脂：{context.body_metrics_summary.get('latest_bodyfat_pct') or '—'} %

### 饮食（报告日）
- 热量：{context.nutrition_summary.get('today_totals', {}).get('calories', 0):.0f} kcal
- 蛋白质：{context.nutrition_summary.get('today_totals', {}).get('protein_g', 0):.0f} g

### 训练（报告日）
{training_detail}

---

_{DISCLAIMER}_
"""
    return header


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


def _load_report_day_training(session: Session, user_id: int, report_date: date) -> list[dict]:
    result = query_training_logs(session, user_id, report_date, report_date)
    return result.get("sessions") or []
