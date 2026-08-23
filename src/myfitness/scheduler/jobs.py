"""调度器执行入口。"""

from __future__ import annotations

import logging

from myfitness.db.repositories.reports import ScheduledTaskRepository
from myfitness.db.session import session_scope
from myfitness.services.daily_report import run_daily_report
from myfitness.sync.orchestrator import run_sync

logger = logging.getLogger(__name__)


def run_scheduled_daily_report(user_id: int, task_id: int) -> None:
    logger.info("执行定时日报 user=%s task=%s", user_id, task_id)
    try:
        with session_scope() as session:
            result = run_daily_report(session, user_id, sync_first=True)
            ScheduledTaskRepository(session, user_id).record_run(task_id, "success")
        logger.info("日报完成: %s", result.get("report_date"))
    except Exception as exc:
        logger.exception("定时日报失败")
        with session_scope() as session:
            ScheduledTaskRepository(session, user_id).record_run(task_id, "failed", str(exc))


def run_scheduled_sync(user_id: int, task_id: int) -> None:
    logger.info("执行定时同步 user=%s task=%s", user_id, task_id)
    try:
        with session_scope() as session:
            run_sync(session, user_id, days=2)
            ScheduledTaskRepository(session, user_id).record_run(task_id, "success")
    except Exception as exc:
        logger.exception("定时同步失败")
        with session_scope() as session:
            ScheduledTaskRepository(session, user_id).record_run(task_id, "failed", str(exc))
