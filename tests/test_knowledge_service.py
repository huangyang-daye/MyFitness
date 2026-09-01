"""知识库服务测试。"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, KnowledgeEntry, User
from myfitness.rag.chunking import entry_to_chunks
from myfitness.rag.knowledge_service import KnowledgeError, create_knowledge


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


def test_entry_to_chunks_splits_long_content():
    entry = KnowledgeEntry(
        id=1,
        user_id=1,
        title="测试",
        content="第一段\n\n" + ("内容" * 400),
    )
    chunks = entry_to_chunks(entry)
    assert len(chunks) >= 1
    assert chunks[0].domain == "knowledge"
    assert chunks[0].source_type == "knowledge"


def test_create_knowledge_validates(db_session):
    with pytest.raises(KnowledgeError, match="标题"):
        create_knowledge(db_session, 1, title="", content="正文")


def test_create_knowledge_persists(db_session, monkeypatch):
    monkeypatch.setattr(
        "myfitness.rag.knowledge_service.index_knowledge_entry",
        lambda *args, **kwargs: {"indexed": 1, "skipped": 0, "failed": 0},
    )
    result = create_knowledge(db_session, 1, title="饮食", content="少油少盐")
    assert result["entry"]["title"] == "饮食"
