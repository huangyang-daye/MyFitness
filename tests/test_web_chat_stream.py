from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.api.asgi_app import create_app
from myfitness.api.web import AgentWebApplication
from myfitness.db.models import Base, User
from myfitness.schemas.state import ChatMessage


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
    return AgentWebApplication(tmp_path, history_dir=tmp_path / "chats")


def _fake_iter_turn(_session, state, message, on_progress=None):
    if on_progress:
        on_progress("识别意图…")
        on_progress("Summary 生成回复中…")
    state.messages.append(ChatMessage(role="user", content=message))

    def chunks():
        yield "你好"
        yield "，"
        yield "世界"

    return state, chunks()


def test_open_does_not_register_empty_session(web_app):
    assert web_app.list_sessions()["sessions"] == []


def test_empty_first_message_does_not_register_session(web_app):
    with pytest.raises(ValueError, match="空"):
        web_app.stream_message(None, "   ")
    assert web_app.list_sessions()["sessions"] == []


def test_first_streamed_message_creates_and_registers_session(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", _fake_iter_turn)
    events: list[tuple[str, dict]] = []

    payload = web_app.stream_message(
        None, "开始对话", emit=lambda name, data: events.append((name, data))
    )

    sessions = web_app.list_sessions()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == payload["session_id"]
    assert sessions[0]["title"] == "开始对话"
    assert [name for name, _ in events][:1] == ["progress"]
    assert [name for name, _ in events if name == "delta"] == ["delta", "delta", "delta"]
    assert "".join(data["text"] for name, data in events if name == "delta") == "你好，世界"
    assert events[-1][0] == "done"
    assert payload["reply"] == "你好，世界"
    saved = web_app.history.load(payload["session_id"])
    assert [item.role for item in saved.messages] == ["user", "assistant"]
    assert saved.messages[-1].content == "你好，世界"


def test_second_streamed_message_reuses_session(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", _fake_iter_turn)
    first = web_app.stream_message(None, "第一句")
    second = web_app.stream_message(first["session_id"], "第二句")

    assert second["session_id"] == first["session_id"]
    assert len(web_app.list_sessions()["sessions"]) == 1
    saved = web_app.history.load(first["session_id"])
    assert [item.content for item in saved.messages if item.role == "user"] == ["第一句", "第二句"]


def test_rule_reply_still_emits_delta_without_creating_session_upfront(web_app, monkeypatch):
    monkeypatch.setattr("myfitness.graph.chat.is_llm_configured", lambda: False)
    events: list[tuple[str, dict]] = []

    assert web_app.list_sessions()["sessions"] == []
    payload = web_app.stream_message(
        None, "你好", emit=lambda name, data: events.append((name, data))
    )

    assert payload["session_id"]
    assert any(name == "session" for name, _ in events)
    assert any(name == "delta" for name, _ in events)
    assert events[-1][0] == "done"
    assert "MyFitness" in payload["reply"]
    assert len(web_app.list_sessions()["sessions"]) == 1


def test_edit_message_interrupts_old_stream_and_reruns_without_deadlock(
    web_app,
    monkeypatch,
):
    """编辑旧消息应先打断原流，再在同一会话锁内重跑且正常结束。"""

    @contextmanager
    def fake_scope():
        yield object()

    stream_started = threading.Event()
    release_old_chunk = threading.Event()

    def fake_iter_turn(_session, state, message, on_progress=None):
        state.user_message = message
        state.messages.append(ChatMessage(role="user", content=message))
        if on_progress:
            on_progress("识别意图…")

        def chunks():
            if message == "原始问题":
                stream_started.set()
                release_old_chunk.wait(timeout=2)
                yield "不应保留的旧回复"
            else:
                yield "编辑后的新回复"

        return state, chunks()

    monkeypatch.setattr("myfitness.api.web.session_scope", fake_scope)
    monkeypatch.setattr("myfitness.api.web.get_or_create_default_user", lambda *_args: None)
    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", fake_iter_turn)

    session_id = web_app.create_session()["session_id"]
    old_abort_evt = web_app._abort_event_for(session_id)
    old_events: list[tuple[str, dict]] = []
    edit_events: list[tuple[str, dict]] = []
    old_results: list[dict] = []
    edit_results: list[dict] = []
    errors: list[Exception] = []

    def run_old_stream():
        try:
            old_results.append(
                web_app.stream_message(
                    session_id,
                    "原始问题",
                    emit=lambda name, data: old_events.append((name, data)),
                )
            )
        except Exception as exc:  # pragma: no cover - 线程错误转交主线程断言
            errors.append(exc)

    def run_edit():
        try:
            edit_results.append(
                web_app.edit_message(
                    session_id,
                    0,
                    "编辑后的问题",
                    emit=lambda name, data: edit_events.append((name, data)),
                )
            )
        except Exception as exc:  # pragma: no cover - 线程错误转交主线程断言
            errors.append(exc)

    old_thread = threading.Thread(target=run_old_stream, daemon=True)
    old_thread.start()
    assert stream_started.wait(timeout=1)

    edit_thread = threading.Thread(target=run_edit, daemon=True)
    edit_thread.start()
    assert old_abort_evt.wait(timeout=1), "编辑请求应优先发送旧流中断信号"
    release_old_chunk.set()

    old_thread.join(timeout=2)
    edit_thread.join(timeout=2)
    assert not old_thread.is_alive()
    assert not edit_thread.is_alive(), "编辑重跑不应重复获取会话锁而死锁"
    assert errors == []
    assert old_results and edit_results
    assert any(name == "aborted" for name, _ in old_events)
    assert edit_events[-1][0] == "done"

    saved = web_app.history.load(session_id)
    assert [item.content for item in saved.messages if item.role == "user"] == [
        "编辑后的问题"
    ]
    assert saved.messages[-1].content == "编辑后的新回复"


def test_fastapi_stream_endpoint_uses_sse(tmp_path, monkeypatch):
    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr("myfitness.api.web.session_scope", fake_scope)
    monkeypatch.setattr("myfitness.api.web.get_or_create_default_user", lambda *_args: None)
    monkeypatch.setattr(
        "myfitness.api.web.get_settings", lambda: SimpleNamespace(default_user_id=1)
    )
    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", _fake_iter_turn)

    web = AgentWebApplication(tmp_path, history_dir=tmp_path / "chats")
    client = TestClient(create_app(web))
    rejected = client.post("/api/sessions/stream", json={"message": "  "})
    assert rejected.status_code == 400
    assert "json" in (rejected.headers.get("content-type") or "")

    with client.stream("POST", "/api/sessions/stream", json={"message": "开始对话"}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        raw = "".join(response.iter_text())
    assert "event: progress" in raw
    assert "event: session" in raw
    assert "event: delta" in raw
    assert "你好" in raw
    assert "event: done" in raw
    assert web.list_sessions()["sessions"]
