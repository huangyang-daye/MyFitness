"""对话进度回调测试。"""

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.db.models import Base, BodyMetric, User
from myfitness.graph.chat import new_chat_state, prepare_chat_turn
from myfitness.graph.progress import label_for


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, name="test")
    session.add(user)
    session.add(
        BodyMetric(
            user_id=1,
            record_date=__import__("datetime").date.today(),
            metric_type="weight",
            value=70.0,
            unit="kg",
            source="manual",
        )
    )
    session.flush()
    yield session
    session.close()


def test_label_for_known_tools():
    assert "身体" in label_for("query_body_metrics")
    assert "Summary" in label_for("summary")


def test_prepare_chat_turn_emits_progress(db_session):
    steps: list[str] = []
    state = new_chat_state(user_id=1)
    with patch("myfitness.graph.chat.is_llm_configured", return_value=False):
        prepare_chat_turn(
            db_session,
            state,
            "昨天体重多少？",
            on_progress=steps.append,
        )

    assert any("识别意图" in s for s in steps)
    assert any("查询身体" in s or "加载上下文" in s for s in steps)
    assert any("Summary" in s or "BodyMonitor" in s for s in steps)
