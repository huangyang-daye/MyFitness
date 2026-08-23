"""APScheduler 定时任务管理。"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from myfitness.config import get_settings
from myfitness.db.repositories.reports import ScheduledTaskRepository
from myfitness.db.session import session_scope
from myfitness.scheduler.jobs import run_scheduled_daily_report, run_scheduled_sync

logger = logging.getLogger(__name__)

_scheduler = None
TZ = ZoneInfo("Asia/Shanghai")


def get_scheduler():
    global _scheduler
    if _scheduler is None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            raise ImportError(
                "定时任务需安装：pip install -e \".[agents]\""
            ) from exc
        _scheduler = BackgroundScheduler(timezone=TZ)
    return _scheduler


def _parse_hm(time_of_day: str) -> tuple[int, int]:
    parts = time_of_day.split(":")
    return int(parts[0]), int(parts[1])


def _job_id(task_id: int) -> str:
    return f"scheduled_task_{task_id}"


def register_task(
    *,
    task_id: int,
    user_id: int,
    task_type: str,
    time_of_day: str,
    label: str,
) -> None:
    from apscheduler.triggers.cron import CronTrigger

    scheduler = get_scheduler()
    job_id = _job_id(task_id)
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    hour, minute = _parse_hm(time_of_day)
    if task_type == ScheduledTaskRepository.TASK_DAILY_REPORT:
        func = run_scheduled_daily_report
    elif task_type == ScheduledTaskRepository.TASK_SYNC:
        func = run_scheduled_sync
    else:
        logger.warning("未知任务类型: %s", task_type)
        return

    scheduler.add_job(
        func,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=TZ),
        id=job_id,
        kwargs={"user_id": user_id, "task_id": task_id},
        replace_existing=True,
        misfire_grace_time=3600,
    )
    logger.info("已注册定时任务 %s @ %s", label, time_of_day)


def reload_scheduler_jobs(user_id: int | None = None) -> int:
    scheduler = get_scheduler()
    settings = get_settings()
    uid = user_id or settings.default_user_id

    for job in scheduler.get_jobs():
        if job.id.startswith("scheduled_task_"):
            scheduler.remove_job(job.id)

    with session_scope() as session:
        tasks = ScheduledTaskRepository(session, uid).list_enabled()
        for task in tasks:
            register_task(
                task_id=task.id,
                user_id=task.user_id,
                task_type=task.task_type,
                time_of_day=task.time_of_day,
                label=task.label,
            )
        return len(tasks)


def start_scheduler(user_id: int | None = None, ensure_defaults: bool = True) -> int:
    """启动调度器并加载 DB 任务；可选从配置种子默认日报任务。"""
    settings = get_settings()
    uid = user_id or settings.default_user_id

    if ensure_defaults:
        _ensure_default_tasks(uid, settings.daily_report_time)

    count = reload_scheduler_jobs(uid)
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("调度器已启动，已加载 %d 个任务", count)
    return count


def stop_scheduler() -> None:
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("调度器已停止")


def scheduler_running() -> bool:
    try:
        return get_scheduler().running
    except Exception:
        return False


def _ensure_default_tasks(user_id: int, daily_report_time: str) -> None:
    with session_scope() as session:
        repo = ScheduledTaskRepository(session, user_id)
        if repo.get_by_type(ScheduledTaskRepository.TASK_DAILY_REPORT) is None:
            repo.upsert(
                ScheduledTaskRepository.TASK_DAILY_REPORT,
                "每日健康日报",
                daily_report_time,
                enabled=True,
            )
