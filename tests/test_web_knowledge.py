"""Web 知识库 API 测试。"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.api.asgi_app import create_app
from myfitness.api.web import AgentWebApplication
from myfitness.db.models import Base, User
from myfitness.rag.knowledge_service import KnowledgeError


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
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
    return AgentWebApplication(tmp_path)


def test_list_knowledge_empty(web_app):
    payload = web_app.list_knowledge()
    assert payload["entry_count"] == 0
    assert payload["entries"] == []


def test_create_and_delete_knowledge(web_app):
    with patch("myfitness.rag.knowledge_service.index_knowledge_entry") as index_mock:
        index_mock.return_value = {"indexed": 1, "skipped": 0, "failed": 0}
        created = web_app.create_knowledge({
            "title": "减脂原则",
            "content": "蛋白质每公斤体重 1.6g 以上",
        })
    assert created["entry"]["title"] == "减脂原则"
    entry_id = created["entry"]["id"]

    listed = web_app.list_knowledge()
    assert listed["entry_count"] == 1

    with patch("myfitness.rag.knowledge_service.delete_knowledge_chunks"):
        with patch("myfitness.rag.knowledge_service.index_knowledge_entry") as index_mock:
            index_mock.return_value = {"indexed": 1, "skipped": 0, "failed": 0}
            updated = web_app.update_knowledge(entry_id, {"content": "更新后的内容"})
    assert "更新后的内容" in updated["entry"]["content"]

    deleted = web_app.delete_knowledge(entry_id)
    assert deleted["deleted_id"] == entry_id
    assert web_app.list_knowledge()["entry_count"] == 0


def test_create_knowledge_rejects_empty_title(web_app):
    with pytest.raises(KnowledgeError, match="标题"):
        web_app.create_knowledge({"title": "  ", "content": "正文"})


def test_parse_knowledge_file_fills_title_and_content(web_app):
    parsed = web_app.parse_knowledge_file(
        "饮食偏好.md",
        "# 偏好\n\n午餐以鸡胸肉为主。".encode(),
    )
    assert parsed["title"] == "饮食偏好"
    assert parsed["format"] == "md"
    assert "鸡胸肉" in parsed["content"]
    assert parsed["truncated"] is False


def test_http_parse_knowledge_file(web_app):
    client = TestClient(create_app(web_app))
    content = "# 蛋白质\n\n每公斤 1.6g\n".encode()
    response = client.post(
        "/api/knowledge/parse",
        files={"file": ("蛋白.md", content, "text/markdown")},
    )
    assert response.status_code == 200, response.text
    payload = response.text
    assert "每公斤 1.6g" in payload
    assert "蛋白" in payload
