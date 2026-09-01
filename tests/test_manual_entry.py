from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myfitness.agents.manual_parser import parse_body_entry, parse_nutrition_entry
from myfitness.db.models import Base, User
from myfitness.db.repositories.metrics import BodyMetricRepository, SOURCE_MANUAL
from myfitness.graph.chat import new_chat_state, run_chat_turn


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


def test_parse_body_entry():
    payload = parse_body_entry("记录体重 72.5kg")
    assert payload
    assert payload["records"][0]["metric_type"] == "weight"
    assert payload["records"][0]["value"] == 72.5


def test_parse_body_entry_with_date_prefix():
    payload = parse_body_entry("以2025年9月1日为起点，记录我的初始体重为130kg，初始体脂率为37%")
    assert payload
    records = {item["metric_type"]: item for item in payload["records"]}
    assert records["weight"]["value"] == 130.0
    assert records["bodyfat"]["value"] == 37.0
    assert records["weight"]["record_date"] == "2025-09-01"


def test_parse_nutrition_entry():
    payload = parse_nutrition_entry("午餐 鸡胸肉 200g 苹果 1个")
    assert payload
    assert len(payload["items"]) == 2


def test_manual_entry_confirmation_flow(db_session):
    state = new_chat_state(user_id=1)
    state = run_chat_turn(db_session, state, "记录体重 73kg")
    assert state.pending_confirmation is not None

    state = run_chat_turn(db_session, state, "确认")
    repo = BodyMetricRepository(db_session, 1)
    metric = repo.get_effective_value(date.today(), "weight")
    assert metric is not None
    assert float(metric.value) == 73.0
    assert metric.source == SOURCE_MANUAL


def test_manual_entry_cancel(db_session):
    state = new_chat_state(user_id=1)
    state = run_chat_turn(db_session, state, "午餐 鸡蛋 2个")
    assert state.pending_confirmation

    state = run_chat_turn(db_session, state, "取消")
    assert state.pending_confirmation is None
    assert "取消" in state.reply
