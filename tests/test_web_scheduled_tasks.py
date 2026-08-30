from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.api.web import AgentWebApplication, ScheduledTaskNotFound
from myfitness.db.models import Base, ScheduledTask, User


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.add(
        ScheduledTask(
            user_id=1,
            task_type="daily_report",
            label="每日健康日报",
            time_of_day="07:00",
            enabled=True,
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.fixture
def web_app(tmp_path, db_session, monkeypatch):
    @contextmanager
    def fake_scope():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    monkeypatch.setattr("myfitness.api.web.session_scope", fake_scope)
    monkeypatch.setattr(
        "myfitness.api.web.get_settings", lambda: SimpleNamespace(default_user_id=1)
    )
    monkeypatch.setattr(AgentWebApplication, "_scheduler_running", staticmethod(lambda: True))
    return AgentWebApplication(tmp_path)


def test_lists_tasks_for_web_ui(web_app):
    payload = web_app.list_scheduled_tasks()

    assert payload["scheduler_running"] is True
    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["tasks"][0]["label"] == "每日健康日报"
    assert payload["tasks"][0]["content_label"] == "生成健康日报"


def test_updates_task_and_reloads_scheduler(web_app, db_session):
    task_id = db_session.query(ScheduledTask).one().id
    with patch("myfitness.scheduler.manager.reload_scheduler_jobs", return_value=1) as reload:
        payload = web_app.update_scheduled_task(
            task_id,
            {
                "label": "晚间健康日报",
                "time_of_day": "21:30",
                "enabled": False,
            },
        )

    db_session.expire_all()
    task = db_session.get(ScheduledTask, task_id)
    assert task.label == "晚间健康日报"
    assert task.time_of_day == "21:30"
    assert task.enabled is False
    assert payload["task"]["enabled"] is False
    reload.assert_called_once_with(1)


@pytest.mark.parametrize("bad_time", ["7:00", "24:00", "12:60", "noon"])
def test_rejects_invalid_task_time(web_app, db_session, bad_time):
    task_id = db_session.query(ScheduledTask).one().id
    with pytest.raises(ValueError, match="HH:MM"):
        web_app.update_scheduled_task(task_id, {"time_of_day": bad_time})


def test_cannot_update_another_users_or_missing_task(web_app):
    with pytest.raises(ScheduledTaskNotFound):
        web_app.update_scheduled_task(9999, {"enabled": False})


def test_rejects_unknown_task_content(web_app, db_session):
    task_id = db_session.query(ScheduledTask).one().id
    with pytest.raises(ValueError, match="任务内容"):
        web_app.update_scheduled_task(task_id, {"task_type": "shell_command"})
