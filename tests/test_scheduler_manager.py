"""调度器管理测试。"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, User
from myfitness.db.repositories.reports import ScheduledTaskRepository
from myfitness.scheduler.manager import get_scheduler, reload_scheduler_jobs, register_task


@pytest.fixture(autouse=True)
def reset_scheduler():
    import myfitness.scheduler.manager as mgr

    old = mgr._scheduler
    mgr._scheduler = None
    yield
    sched = mgr._scheduler
    if sched is not None and sched.running:
        sched.shutdown(wait=False)
    mgr._scheduler = old


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.flush()
    yield session
    session.close()


def test_register_task_with_plain_fields():
    register_task(
        task_id=1,
        user_id=1,
        task_type=ScheduledTaskRepository.TASK_DAILY_REPORT,
        time_of_day="07:00",
        label="每日健康日报",
    )
    job = get_scheduler().get_job("scheduled_task_1")
    assert job is not None
    assert job.kwargs == {"user_id": 1, "task_id": 1}


def test_reload_scheduler_jobs_uses_db_tasks(db_session):
    repo = ScheduledTaskRepository(db_session, 1)
    task = repo.upsert(
        ScheduledTaskRepository.TASK_DAILY_REPORT,
        "每日健康日报",
        "08:30",
        enabled=True,
    )
    db_session.commit()

    with patch("myfitness.scheduler.manager.session_scope") as scope_mock:
        scope_mock.return_value.__enter__.return_value = db_session
        scope_mock.return_value.__exit__.return_value = False
        count = reload_scheduler_jobs(user_id=1)

    assert count == 1
    job = get_scheduler().get_job(f"scheduled_task_{task.id}")
    assert job is not None
    assert job.kwargs["task_id"] == task.id
