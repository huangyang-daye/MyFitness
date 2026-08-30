"""定时任务写入工具。

均用 LangChain `@tool` 修饰；`session` / `user_id` 通过 `InjectedToolArg` 注入。
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from myfitness.db.repositories.reports import ScheduledTaskRepository


@tool
def apply_schedule_upsert(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    payload: dict,
) -> str:
    """新建或更新一个每日定时任务（如日报生成）。

    Args:
        payload: 形如 {"task_type": "daily_report", "label": "每日健康日报",
            "time_of_day": "07:00", "enabled": true}。
    """
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


@tool
def apply_schedule_cancel(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
    task_type: str,
) -> str:
    """停用某个类型的定时任务。

    Args:
        task_type: 任务类型，如 daily_report / sync。
    """
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


@tool
def list_scheduled_tasks(
    session: Annotated[Session, InjectedToolArg],
    user_id: Annotated[int, InjectedToolArg],
) -> list[dict]:
    """列出当前用户的所有定时任务（含类型、标签、执行时间、是否启用）。"""
    rows = ScheduledTaskRepository(session, user_id).list_all()
    return [
        {
            "task_type": r.task_type,
            "label": r.label,
            "time_of_day": r.time_of_day,
            "enabled": r.enabled,
            "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
        }
        for r in rows
    ]
