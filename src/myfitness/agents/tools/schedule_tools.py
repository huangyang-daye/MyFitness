"""定时任务写入工具。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from myfitness.db.repositories.reports import ScheduledTaskRepository


def apply_schedule_upsert(session: Session, user_id: int, payload: dict) -> str:
    repo = ScheduledTaskRepository(session, user_id)
    task = repo.upsert(
        task_type=payload["task_type"],
        label=payload.get("label") or payload["task_type"],
        time_of_day=payload["time_of_day"],
        enabled=payload.get("enabled", True),
    )
    try:
        from myfitness.scheduler.manager import reload_scheduler_jobs

        reload_scheduler_jobs()
    except Exception:
        pass
    return f"已保存定时任务：{task.label}，每天 {task.time_of_day}"


def apply_schedule_cancel(session: Session, user_id: int, task_type: str) -> str:
    repo = ScheduledTaskRepository(session, user_id)
    task = repo.disable(task_type)
    if not task:
        return f"未找到类型为 {task_type} 的定时任务。"
    try:
        from myfitness.scheduler.manager import reload_scheduler_jobs

        reload_scheduler_jobs()
    except Exception:
        pass
    label = task.label
    return f"已停用定时任务：{label}"


def list_scheduled_tasks(session: Session, user_id: int) -> list:
    return ScheduledTaskRepository(session, user_id).list_all()
