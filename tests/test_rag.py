"""RAG 模块测试。"""

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.config import Settings
from myfitness.db.models import Base, BodyMetric, User
from myfitness.rag.chunking import collect_chunks, content_hash
from myfitness.rag.dimensions import parse_vector_column_type
from myfitness.rag.dimensions import parse_vector_column_type
from myfitness.rag.embedding import (
    EmbeddingError,
    embedding_host_supported,
    embed_texts,
    is_embedding_configured,
)
from myfitness.rag.format import format_retrieved_chunks
from myfitness.rag.retriever import should_retrieve
from myfitness.rag.schemas import RetrievedChunk
from myfitness.rag.store import search_chunks
from myfitness.schemas.state import Intent


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(User(id=1, name="test"))
    session.add(
        BodyMetric(
            user_id=1,
            record_date=date(2026, 8, 20),
            metric_type="weight",
            value=72.5,
            unit="kg",
            source="manual",
        )
    )
    session.flush()
    yield session
    session.close()


def test_content_hash_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abcd")


def test_collect_body_chunks(db_session):
    chunks = collect_chunks(db_session, 1, start_date=date(2026, 8, 20), end_date=date(2026, 8, 20))
    assert len(chunks) == 1
    assert chunks[0].domain == "body"
    assert "72.5" in chunks[0].content


def test_collect_training_chunks(db_session):
    from myfitness.db.models import TrainingLog

    db_session.add(
        TrainingLog(
            user_id=1,
            record_date=date(2026, 9, 1),
            title="腿臀",
            raw_payload={
                "localid": 1,
                "datestr": "2026-09-01",
                "title": "腿臀",
                "movements": [
                    {
                        "name": "深蹲",
                        "sets": [
                            {"done": True, "weight": "60", "unit": "kg", "reps": "8"},
                        ],
                    }
                ],
            },
            source="xunji",
        )
    )
    db_session.flush()
    chunks = collect_chunks(db_session, 1, start_date=date(2026, 9, 1), end_date=date(2026, 9, 1))
    training = [c for c in chunks if c.domain == "fitness"]
    assert len(training) == 1
    assert "深蹲" in training[0].content
    assert "60kg×8" in training[0].content


def test_should_retrieve_respects_intent():
    with patch("myfitness.rag.retriever.get_settings") as settings_mock:
        settings_mock.return_value.rag_enabled = True
        assert should_retrieve(Intent.TREND_ANALYSIS)
        assert should_retrieve(Intent.WEB_SEARCH)
        assert not should_retrieve(Intent.SYNC_TRIGGER)


def test_format_retrieved_chunks():
    text = format_retrieved_chunks(
        [
            RetrievedChunk(
                id=1,
                source_type="body_daily",
                source_id="2026-08-20",
                domain="body",
                title="2026-08-20 身体数据",
                content="2026-08-20 身体数据：体重 72.5kg",
                record_date=date(2026, 8, 20),
                similarity=0.88,
            )
        ]
    )
    assert "72.5" in text
    assert "0.88" in text


def test_embedding_host_supported_rejects_deepseek():
    assert not embedding_host_supported("https://api.deepseek.com")
    assert not embedding_host_supported("https://api.deepseek.com/v1")
    assert embedding_host_supported("https://api.openai.com/v1")
    assert embedding_host_supported("https://api.siliconflow.cn/v1")


def test_deepseek_llm_is_not_embedding_configured():
    settings = Settings(
        llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-test",
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="text-embedding-3-small",
    )
    assert not is_embedding_configured(settings)


def test_explicit_embedding_url_works_with_deepseek_llm():
    settings = Settings(
        llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-ds",
        embedding_base_url="https://api.openai.com/v1",
        embedding_api_key="sk-oai",
        embedding_model="text-embedding-3-small",
    )
    assert is_embedding_configured(settings)


def test_openai_llm_fallback_is_embedding_configured():
    settings = Settings(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="sk-oai",
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="text-embedding-3-small",
    )
    assert is_embedding_configured(settings)


def test_embed_texts_rejects_deepseek_without_http_call():
    settings = Settings(
        llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-test",
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="text-embedding-3-small",
    )
    with pytest.raises(EmbeddingError, match="没有 /embeddings"):
        embed_texts(["hello"], settings=settings)


def test_search_chunks_swallows_embedding_error(db_session):
    with (
        patch("myfitness.rag.store.rag_is_available", return_value=True),
        patch("myfitness.rag.embedding.embed_text", side_effect=EmbeddingError("boom")),
    ):
        assert search_chunks(db_session, 1, "体重趋势") == []


def test_parse_vector_column_type():
    assert parse_vector_column_type("vector(1536)") == 1536
    assert parse_vector_column_type("vector(1024)") == 1024
    assert parse_vector_column_type("vector") is None
    assert parse_vector_column_type(None) is None
