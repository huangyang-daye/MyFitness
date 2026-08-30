from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.api.web import AgentUiHttpServer, AgentWebApplication
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


def test_http_stream_endpoint_uses_sse(tmp_path, monkeypatch):
    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr("myfitness.api.web.session_scope", fake_scope)
    monkeypatch.setattr("myfitness.api.web.get_or_create_default_user", lambda *_args: None)
    monkeypatch.setattr(
        "myfitness.api.web.get_settings", lambda: SimpleNamespace(default_user_id=1)
    )
    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", _fake_iter_turn)

    app = AgentWebApplication(tmp_path, history_dir=tmp_path / "chats")
    server = AgentUiHttpServer(("127.0.0.1", 0), app)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        empty = json.dumps({"message": "  "}).encode()
        conn.request(
            "POST",
            "/api/sessions/stream",
            body=empty,
            headers={"Content-Type": "application/json", "Content-Length": str(len(empty))},
        )
        rejected = conn.getresponse()
        assert rejected.status == 400
        assert "json" in (rejected.getheader("Content-Type") or "")
        assert rejected.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        body = json.dumps({"message": "开始对话"}).encode()
        conn.request(
            "POST",
            "/api/sessions/stream",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = conn.getresponse()
        assert response.status == 200
        assert "text/event-stream" in (response.getheader("Content-Type") or "")
        raw = response.read().decode("utf-8")
        assert "event: progress" in raw
        assert "event: session" in raw
        assert "event: delta" in raw
        assert "你好" in raw
        assert "event: done" in raw
        conn.close()
        assert app.list_sessions()["sessions"]
    finally:
        server.shutdown()
        server.server_close()
