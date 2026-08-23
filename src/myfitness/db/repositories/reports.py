"""daily_reports 与 scheduled_tasks 仓储。"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from myfitness.db.models import DailyReport, ScheduledTask


class DailyReportRepository:
    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def get_by_date(self, report_date: date) -> DailyReport | None:
        return self.session.scalar(
            select(DailyReport).where(
                DailyReport.user_id == self.user_id,
                DailyReport.report_date == report_date,
            )
        )

    def upsert(self, report_date: date, content_md: str, agent_outputs: dict | None = None) -> DailyReport:
        existing = self.get_by_date(report_date)
        if existing:
            existing.content_md = content_md
            existing.agent_outputs = agent_outputs
            report = existing
        else:
            report = DailyReport(
                user_id=self.user_id,
                report_date=report_date,
                content_md=content_md,
                agent_outputs=agent_outputs,
            )
            self.session.add(report)
        self.session.flush()
        return report

    def list_recent(self, limit: int = 7) -> list[DailyReport]:
        stmt = (
            select(DailyReport)
            .where(DailyReport.user_id == self.user_id)
            .order_by(DailyReport.report_date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())


class ScheduledTaskRepository:
    TASK_DAILY_REPORT = "daily_report"
    TASK_SYNC = "sync"

    def __init__(self, session: Session, user_id: int):
        self.session = session
        self.user_id = user_id

    def list_all(self) -> list[ScheduledTask]:
        stmt = (
            select(ScheduledTask)
            .where(ScheduledTask.user_id == self.user_id)
            .order_by(ScheduledTask.task_type)
        )
        return list(self.session.scalars(stmt).all())

    def list_enabled(self) -> list[ScheduledTask]:
        stmt = (
            select(ScheduledTask)
            .where(ScheduledTask.user_id == self.user_id, ScheduledTask.enabled.is_(True))
            .order_by(ScheduledTask.task_type)
        )
        return list(self.session.scalars(stmt).all())

    def get_by_type(self, task_type: str) -> ScheduledTask | None:
        return self.session.scalar(
            select(ScheduledTask).where(
                ScheduledTask.user_id == self.user_id,
                ScheduledTask.task_type == task_type,
            )
        )

    def upsert(
        self,
        task_type: str,
        label: str,
        time_of_day: str,
        enabled: bool = True,
    ) -> ScheduledTask:
        existing = self.get_by_type(task_type)
        if existing:
            existing.label = label
            existing.time_of_day = time_of_day
            existing.enabled = enabled
            task = existing
        else:
            task = ScheduledTask(
                user_id=self.user_id,
                task_type=task_type,
                label=label,
                time_of_day=time_of_day,
                enabled=enabled,
            )
            self.session.add(task)
        self.session.flush()
        return task

    def disable(self, task_type: str) -> ScheduledTask | None:
        task = self.get_by_type(task_type)
        if not task:
            return None
        task.enabled = False
        self.session.flush()
        return task

    def record_run(self, task_id: int, status: str, error: str | None = None) -> None:
        task = self.session.get(ScheduledTask, task_id)
        if not task:
            return
        task.last_run_at = datetime.now(UTC)
        task.last_status = status
        task.last_error = error
