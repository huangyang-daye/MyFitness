"""断连取消 LLM 与 Redis checkpointer 相关测试。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from myfitness.api.web import AgentWebApplication
from myfitness.graph.abort import abort_scope, is_aborted
from myfitness.llm.factory import stream_chat_completion
from myfitness.schemas.state import ChatMessage


def test_abort_scope_flags_is_aborted():
    evt = threading.Event()
    assert is_aborted() is False
    with abort_scope(evt.is_set):
        assert is_aborted() is False
        evt.set()
        assert is_aborted() is True
    assert is_aborted() is False


def test_stream_chat_completion_stops_when_aborted(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"A"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"B"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"C"}}]}'

        def close(self):
            self.closed = True

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)
    monkeypatch.setattr(
        "myfitness.llm.factory.get_llm_config",
        lambda settings=None: SimpleNamespace(
            model="m",
            temperature=0,
            max_tokens=None,
            timeout=5,
            api_key="k",
            chat_completions_url="http://example/v1/chat/completions",
        ),
    )
    guard = MagicMock()
    monkeypatch.setattr("myfitness.llm.factory.get_llm_guard", lambda: guard)

    stop = threading.Event()
    chunks: list[str] = []
    for i, token in enumerate(
        stream_chat_completion([{"role": "user", "content": "hi"}], should_abort=stop.is_set)
    ):
        chunks.append(token)
        if i == 0:
            stop.set()
    assert chunks == ["A"]


def test_client_gone_saves_partial_and_marks_incomplete(tmp_path, monkeypatch):
    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr("myfitness.api.web.session_scope", fake_scope)
    monkeypatch.setattr("myfitness.api.web.get_or_create_default_user", lambda *_a: None)
    monkeypatch.setattr(
        "myfitness.api.web.get_settings", lambda: SimpleNamespace(default_user_id=1)
    )

    release = threading.Event()
    client_gone = threading.Event()

    def fake_iter(_session, state, message, on_progress=None):
        state.messages.append(ChatMessage(role="user", content=message))
        if on_progress:
            on_progress("识别意图…")

        def chunks():
            yield "部分"
            release.wait(timeout=2)
            yield "不应发出"

        return state, chunks()

    monkeypatch.setattr("myfitness.api.web.iter_chat_turn", fake_iter)
    app = AgentWebApplication(tmp_path, history_dir=tmp_path / "chats")
    events: list[tuple[str, dict]] = []

    def emit(name, data):
        events.append((name, data))
        if name == "delta":
            client_gone.set()
            release.set()

    payload = app.stream_message(None, "测试断开", emit=emit, client_gone=client_gone)
    assert payload["session_id"]
    assert any(name == "aborted" for name, _ in events)
    saved = app.history.load(payload["session_id"])
    assert saved.metadata.graph_incomplete is True
    assert "部分" in saved.reply
    assert "继续" in saved.reply


def test_redis_checkpointer_roundtrip(monkeypatch):
    store: dict[bytes, bytes] = {}

    class FakeRedis:
        def ping(self):
            return True

        def get(self, key):
            return store.get(key if isinstance(key, bytes) else key.encode())

        def set(self, key, value, ex=None):
            k = key if isinstance(key, bytes) else key.encode()
            store[k] = value
            return True

        def delete(self, key):
            k = key if isinstance(key, bytes) else key.encode()
            store.pop(k, None)
            return 1

    class FakeRedisCls:
        @staticmethod
        def from_url(*_a, **_k):
            return FakeRedis()

    import sys
    import types

    fake_mod = types.ModuleType("redis")
    fake_mod.Redis = FakeRedisCls
    monkeypatch.setitem(sys.modules, "redis", fake_mod)

    from myfitness.graph import langgraph_flow as lg
    from myfitness.graph.redis_checkpointer import RedisCheckpointSaver

    lg.reset_checkpointer_cache()
    saver = RedisCheckpointSaver("redis://fake", ttl_seconds=60)
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": "1ef00000-0000-4000-8000-000000000001",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"judge_attempt": 1},
        "channel_versions": {"judge_attempt": 1},
        "versions_seen": {},
        "updated_channels": None,
    }
    saver.put(config, checkpoint, {"source": "loop", "step": 0, "parents": {}}, {"judge_attempt": 1})
    # 模拟新进程：清空内存再 hydrate
    saver.storage.clear()
    saver.writes.clear()
    saver.blobs.clear()
    saver._hydrated.clear()
    loaded = saver.get_tuple({"configurable": {"thread_id": "t1", "checkpoint_ns": ""}})
    assert loaded is not None
    assert loaded.checkpoint["channel_values"]["judge_attempt"] == 1
    saver.delete_thread("t1")
    assert saver.get_tuple({"configurable": {"thread_id": "t1"}}) is None
    lg.reset_checkpointer_cache()
