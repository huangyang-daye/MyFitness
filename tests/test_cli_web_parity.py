from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace

from rich.console import Console
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from myfitness.api import cli
from myfitness.api.cli import _handle_chat_command, app
from myfitness.chat_history import ChatHistoryStore
from myfitness.config import Settings
from myfitness.db.models import Base, ScheduledTask, User
from myfitness.graph.chat import new_chat_state
from myfitness.llm.registry import get_registry
from myfitness.schemas.state import ChatMessage

runner = CliRunner()


def test_cli_manages_same_model_registry_as_web(monkeypatch):
    result = runner.invoke(
        app,
        [
            "llm",
            "add",
            "--name",
            "Test Model",
            "--base-url",
            "https://api.example.com/v1",
            "--model",
            "model-v1",
            "--api-key",
            "sk-secret1234",
            "--activate",
        ],
    )
    assert result.exit_code == 0, result.output

    preset = get_registry().models()[0]
    assert get_registry().active_id() == preset.id
    assert preset.api_key == "sk-secret1234"

    result = runner.invoke(
        app,
        ["llm", "edit", preset.id, "--name", "Renamed", "--temperature", "0.2"],
    )
    assert result.exit_code == 0, result.output
    assert get_registry().get(preset.id).name == "Renamed"
    assert get_registry().get(preset.id).temperature == 0.2

    captured = {}

    def fake_probe(config, prompt):
        captured["config"] = config
        captured["prompt"] = prompt
        return {"model": config.model, "reply": "OK", "usage": {}, "latency_ms": 1}

    monkeypatch.setattr("myfitness.api.cli.probe_llm_config", fake_probe)
    result = runner.invoke(app, ["llm", "test", "--id", preset.id, "--prompt", "ping"])
    assert result.exit_code == 0, result.output
    assert captured["config"].api_key == "sk-secret1234"
    assert captured["prompt"] == "ping"

    result = runner.invoke(app, ["llm", "delete", preset.id, "--yes"])
    assert result.exit_code == 0, result.output
    assert get_registry().get(preset.id) is None


def test_cli_lists_and_shows_web_chat_history(tmp_path, monkeypatch):
    store = ChatHistoryStore(tmp_path / "history")
    state = new_chat_state()
    state.messages.extend(
        [
            ChatMessage(role="user", content="最近体重如何？"),
            ChatMessage(role="assistant", content="## 结论\n\n整体稳定。"),
        ]
    )
    store.save(state)
    monkeypatch.setattr("myfitness.api.cli.ChatHistoryStore", lambda: store)

    listed = runner.invoke(app, ["session", "list"], terminal_width=180)
    assert listed.exit_code == 0, listed.output
    assert state.session_id in listed.output

    shown = runner.invoke(app, ["session", "show", state.session_id, "--raw"])
    assert shown.exit_code == 0, shown.output
    assert "整体稳定" in shown.output


def test_cli_artifact_show_uses_data_dir_boundary(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "2026-08-29.md"
    report.parent.mkdir(parents=True)
    report.write_text("# 日报\n\n状态良好。", encoding="utf-8")
    monkeypatch.setattr(
        "myfitness.services.artifacts.get_settings",
        lambda: Settings(data_dir=str(data_dir)),
    )

    shown = runner.invoke(app, ["artifact", "show", str(report), "--raw"])
    assert shown.exit_code == 0, shown.output
    assert "状态良好" in shown.output

    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    rejected = runner.invoke(app, ["artifact", "show", str(outside)])
    assert rejected.exit_code == 1
    assert "数据目录" in rejected.output


def test_cli_edits_and_toggles_scheduled_tasks(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'cli.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session.begin() as session:
        session.add(User(id=1, name="test"))

    @contextmanager
    def fake_scope():
        session = Session()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    reloads = []
    monkeypatch.setattr("myfitness.api.cli.session_scope", fake_scope)
    monkeypatch.setattr(
        "myfitness.api.cli.get_settings",
        lambda: SimpleNamespace(default_user_id=1, log_level="INFO"),
    )
    monkeypatch.setattr("myfitness.api.cli._reload_scheduler_after_change", reloads.append)

    added = runner.invoke(
        app,
        [
            "scheduler",
            "add",
            "daily_report",
            "--time",
            "21:30",
            "--label",
            "晚间日报",
        ],
    )
    assert added.exit_code == 0, added.output
    with Session() as session:
        task = session.query(ScheduledTask).one()
        task_id = task.id
        assert task.label == "晚间日报"
        assert task.time_of_day == "21:30"

    edited = runner.invoke(
        app,
        ["scheduler", "edit", str(task_id), "--time", "22:00", "--disabled"],
    )
    assert edited.exit_code == 0, edited.output
    with Session() as session:
        task = session.get(ScheduledTask, task_id)
        assert task.time_of_day == "22:00"
        assert task.enabled is False

    enabled = runner.invoke(app, ["scheduler", "enable", str(task_id)])
    assert enabled.exit_code == 0, enabled.output
    with Session() as session:
        assert session.get(ScheduledTask, task_id).enabled is True
    assert reloads == [1, 1, 1]


def test_cli_rejects_invalid_schedule_time(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "myfitness.api.cli.get_settings",
        lambda: SimpleNamespace(default_user_id=1, log_level="INFO"),
    )
    result = runner.invoke(
        app,
        ["scheduler", "add", "sync", "--time", "7:00"],
    )
    assert result.exit_code == 1
    assert "HH:MM" in result.output


def test_chat_model_command_switches_model_by_number(tmp_path, monkeypatch):
    registry = get_registry()
    first = registry.upsert(
        {
            "name": "First",
            "base_url": "https://first.example/v1",
            "model": "first-model",
            "api_key": "sk-first",
        }
    )
    second = registry.upsert(
        {
            "name": "Second",
            "base_url": "https://second.example/v1",
            "model": "second-model",
            "api_key": "sk-second",
        }
    )
    registry.set_active(first.id)
    models = registry.all_models()
    second_number = models.index(second) + 1
    monkeypatch.setattr("myfitness.api.cli.console.input", lambda _prompt: str(second_number))
    history = ChatHistoryStore(tmp_path / "history")
    state = new_chat_state()

    handled, returned = _handle_chat_command("/model", history, state)

    assert handled is True
    assert returned is state
    assert registry.active_id() == second.id


def test_chat_resume_command_restores_selected_session(tmp_path, monkeypatch):
    history = ChatHistoryStore(tmp_path / "history")
    current = new_chat_state()
    current.messages.append(ChatMessage(role="user", content="当前会话"))
    history.save(current)
    target = new_chat_state()
    target.messages.append(ChatMessage(role="user", content="需要恢复的会话"))
    history.save(target)
    sessions = history.list_sessions()
    target_number = next(
        index
        for index, item in enumerate(sessions, start=1)
        if item.session_id == target.session_id
    )
    monkeypatch.setattr("myfitness.api.cli.console.input", lambda _prompt: str(target_number))

    handled, restored = _handle_chat_command("/resume", history, current)

    assert handled is True
    assert restored.session_id == target.session_id
    assert restored.messages[0].content == "需要恢复的会话"


def test_unknown_chat_slash_command_is_left_for_agent(tmp_path):
    history = ChatHistoryStore(tmp_path / "history")
    state = new_chat_state()

    handled, returned = _handle_chat_command("/unknown", history, state)

    assert handled is False
    assert returned is state


def test_menu_focus_moves_with_arrows_and_wraps():
    assert cli._move_menu_focus(0, "down", 3) == 1
    assert cli._move_menu_focus(0, "up", 3) == 2
    assert cli._move_menu_focus(2, "down", 3) == 0
    assert cli._move_menu_focus(1, "other", 3) == 1


def test_interactive_menu_uses_arrows_and_enter(monkeypatch):
    keys = iter(["down", "down", "up", "enter"])

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            self.updates = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, renderable, refresh=False):
            self.updates.append((renderable, refresh))

    monkeypatch.setattr(cli, "_supports_arrow_menu", lambda: True)
    monkeypatch.setattr(cli, "_read_menu_key", lambda: next(keys))
    monkeypatch.setattr(cli, "Live", FakeLive)

    selected = cli._select_option(
        "选择",
        [("a", "A"), ("b", "B"), ("c", "C")],
        current_value="a",
    )

    assert selected == "b"


def test_home_page_contains_layout_help_model_and_input(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, width=120, color_system=None, force_terminal=False),
    )

    cli._render_chat_home()

    rendered = output.getvalue()
    assert "MYFITNESS" in rendered
    assert "/model" in rendered
    assert "/resume" in rendered
    assert "MODEL" in rendered
    assert "Planner" in rendered
    assert "██" in rendered


def test_resumed_conversation_prints_complete_history(tmp_path, monkeypatch):
    history = ChatHistoryStore(tmp_path / "history")
    target = new_chat_state()
    target.messages.extend(
        [
            ChatMessage(role="user", content="第一条问题"),
            ChatMessage(role="assistant", content="第一条回答"),
            ChatMessage(role="user", content="第二条问题"),
            ChatMessage(role="assistant", content="第二条回答"),
        ]
    )
    history.save(target)
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, width=120, color_system=None, force_terminal=False),
    )
    monkeypatch.setattr(cli, "_select_option", lambda *_args, **_kwargs: target.session_id)

    restored = cli._chat_resume_session(history, None)

    assert restored.session_id == target.session_id
    rendered = output.getvalue()
    for content in ("第一条问题", "第一条回答", "第二条问题", "第二条回答"):
        assert content in rendered


def test_chat_exit_from_home_does_not_create_empty_session(monkeypatch):
    class RecordingHistory:
        def __init__(self):
            self.saved = []

        def save(self, state):
            self.saved.append(state.session_id)

    history = RecordingHistory()
    output = StringIO()
    test_console = Console(file=output, width=120, color_system=None, force_terminal=False)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr(test_console, "input", lambda _prompt: "exit")
    monkeypatch.setattr(cli, "ChatHistoryStore", lambda: history)
    monkeypatch.setattr(cli, "_warmup_database", lambda _user_id: None)
    monkeypatch.setattr(
        cli,
        "_run_llm_warmup",
        lambda: cli.LlmWarmupResult(configured=False, loaded=True),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(default_user_id=1))

    cli.chat(once=False, message=None, no_stream=False, session_id=None)

    assert history.saved == []


def test_first_submitted_message_creates_and_registers_session(monkeypatch):
    class RecordingHistory:
        def __init__(self):
            self.saved_message_counts = []

        def save(self, state):
            self.saved_message_counts.append(len(state.messages))

    @contextmanager
    def fake_scope():
        yield object()

    def fake_turn(_session, state, message, on_progress=None):
        state.messages.append(ChatMessage(role="user", content=message))
        state.reply = "已收到"
        return state

    history = RecordingHistory()
    output = StringIO()
    test_console = Console(file=output, width=120, color_system=None, force_terminal=False)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr(cli, "ChatHistoryStore", lambda: history)
    monkeypatch.setattr(cli, "_warmup_database", lambda _user_id: None)
    monkeypatch.setattr(
        cli,
        "_run_llm_warmup",
        lambda: cli.LlmWarmupResult(configured=False, loaded=True),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(default_user_id=1))
    monkeypatch.setattr(cli, "session_scope", fake_scope)
    monkeypatch.setattr(cli, "get_or_create_default_user", lambda *_args: None)
    monkeypatch.setattr(cli, "run_chat_turn", fake_turn)

    cli.chat(once=False, message="开始对话", no_stream=True, session_id=None)

    assert history.saved_message_counts == [0, 1]
