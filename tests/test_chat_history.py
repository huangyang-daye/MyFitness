import json
import uuid
from datetime import UTC, datetime

import pytest

from myfitness.chat_history import ChatHistoryError, ChatHistoryStore, ChatSessionNotFound
from myfitness.config import get_settings
from myfitness.graph.chat import new_chat_state
from myfitness.schemas.state import ChatMessage


def test_save_creates_one_uuid_named_json_and_loads_state(tmp_path):
    store = ChatHistoryStore(tmp_path)
    state = new_chat_state(user_id=7)
    state.messages.append(
        ChatMessage(role="user", content="分析最近七天体重", timestamp=datetime.now(UTC))
    )

    path = store.save(state)

    assert path == tmp_path / f"{state.session_id}.json"
    assert uuid.UUID(path.stem).version == 4
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["session_id"] == state.session_id
    assert document["title"] == "分析最近七天体重"
    restored = store.load(state.session_id)
    assert restored.user_id == 7
    assert restored.messages[0].content == "分析最近七天体重"


def test_save_updates_same_document_and_preserves_created_at(tmp_path):
    store = ChatHistoryStore(tmp_path)
    state = new_chat_state()
    path = store.save(state)
    first = json.loads(path.read_text(encoding="utf-8"))
    state.messages.append(ChatMessage(role="user", content="第二次写入"))

    store.save(state)

    second = json.loads(path.read_text(encoding="utf-8"))
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert second["created_at"] == first["created_at"]
    assert second["state"]["messages"][0]["content"] == "第二次写入"


def test_list_sessions_is_most_recent_first_and_skips_damage(tmp_path):
    store = ChatHistoryStore(tmp_path)
    older = new_chat_state()
    newer = new_chat_state()
    store.save(older)
    store.save(newer)
    bad = tmp_path / f"{uuid.uuid4()}.json"
    bad.write_text("not json", encoding="utf-8")

    sessions = store.list_sessions()

    assert {item.session_id for item in sessions} == {older.session_id, newer.session_id}
    assert sessions[0].updated_at >= sessions[1].updated_at


def test_rejects_non_uuid_and_missing_session(tmp_path):
    store = ChatHistoryStore(tmp_path)
    with pytest.raises(ChatHistoryError):
        store.load("../../.env")
    with pytest.raises(ChatSessionNotFound):
        store.load(str(uuid.uuid4()))


def test_default_history_dir_comes_from_settings(tmp_path, monkeypatch):
    """对话记录默认落在配置的 chat_history_dir，而不是项目目录。"""
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(tmp_path / "chat-history"))
    get_settings.cache_clear()
    try:
        store = ChatHistoryStore()
    finally:
        get_settings.cache_clear()

    assert store.history_dir == (tmp_path / "chat-history").resolve()
    assert ".chatHistory" not in store.history_dir.parts


def test_rejects_filename_document_id_mismatch(tmp_path):
    store = ChatHistoryStore(tmp_path)
    state = new_chat_state()
    path = store.save(state)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["session_id"] = str(uuid.uuid4())
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ChatHistoryError):
        store.load(state.session_id)
