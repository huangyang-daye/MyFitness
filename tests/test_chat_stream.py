"""Summary LLM 兜底测试。"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.summary import iter_summary_reply, should_stream_summary
from myfitness.db.models import Base, User
from myfitness.graph.chat import finalize_streamed_reply, iter_chat_reply, new_chat_state, prepare_chat_turn
from myfitness.schemas.agent_outputs import AgentOutputs
from myfitness.schemas.state import Intent


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, name="test")
    session.add(user)
    session.flush()
    yield session
    session.close()


def test_should_stream_summary_excludes_fixed_intents():
    assert not should_stream_summary(Intent.MANUAL_ENTRY)
    assert not should_stream_summary(Intent.SYNC_TRIGGER)
    assert should_stream_summary(Intent.DATA_QUERY)


def test_iter_summary_reply_fallback_without_llm():
    with patch("myfitness.agents.summary.is_llm_configured", return_value=False):
        chunks = list(iter_summary_reply(AgentOutputs(), None, Intent.GENERAL, "你好"))
    assert len(chunks) == 1
    assert "MyFitness" in chunks[0]


def test_iter_summary_reply_falls_back_on_llm_error():
    def _raise(*args, **kwargs):
        raise ConnectionError("LLM down")

    with (
        patch("myfitness.agents.summary.is_llm_configured", return_value=True),
        patch("myfitness.agents.summary.stream_chat_completion", side_effect=_raise),
    ):
        chunks = list(iter_summary_reply(AgentOutputs(), None, Intent.GENERAL, "你好"))
    assert len(chunks) == 1
    assert "MyFitness" in chunks[0]


def test_iter_summary_reply_partial_output_appends_note():
    def _partial(*args, **kwargs):
        yield "部分内容"
        raise ConnectionError("中断")

    with (
        patch("myfitness.agents.summary.is_llm_configured", return_value=True),
        patch("myfitness.agents.summary.stream_chat_completion", side_effect=_partial),
    ):
        chunks = list(iter_summary_reply(AgentOutputs(), None, Intent.DATA_QUERY, "近7天体重"))
    text = "".join(chunks)
    assert "部分内容" in text
    assert "不完整" in text


def test_iter_chat_reply_no_disclaimer(db_session):
    state = new_chat_state(user_id=1)
    with (
        patch("myfitness.graph.chat.is_llm_configured", return_value=False),
        patch("myfitness.agents.summary.is_llm_configured", return_value=False),
    ):
        result = prepare_chat_turn(db_session, state, "你好")
    chunks = list(iter_chat_reply(result.state))
    text = "".join(chunks)
    assert "不构成医疗建议" not in text
    assert len(text.strip()) > 0


def test_finalize_streamed_reply(db_session):
    state = new_chat_state(user_id=1)
    finalize_streamed_reply(state, "测试回复")
    assert state.reply == "测试回复"
    assert state.messages[-1].content == "测试回复"
